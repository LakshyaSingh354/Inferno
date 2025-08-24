import hashlib
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import List, Dict, Any
import textwrap
from graph_creation import Graph, TorchFXParser
from graph_optimization import Optimizer


# ================================================================================
# SECTION 1: The Backend - Code Generator
# ================================================================================

class CodeGenerator:
    """
    Takes an optimized Inferno IR Graph and generates compilable C++/CUDA source code.
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        self.template = textwrap.dedent("""\
            #include <torch/extension.h>
            #include <cublas_v2.h>
            #include <vector>

            // Forward declaration of our actual fused kernel implementation
            void fused_gemm_relu_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor);

            // External declaration of cuBLAS handle (defined in fused kernel)
            // extern cublasHandle_t get_cublas_handle();

            // The main forward function for the compiled model
            torch::Tensor {graph_name}_forward(
                {function_args}
            ) {{
                //cublasHandle_t handle = get_cublas_handle();
                {tensor_declarations}
                {kernel_calls}
                return {output_name};
            }}

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
                m.def("forward", &{graph_name}_forward, "Inferno compiled forward pass for {graph_name}");
            }}
        """)

    def generate(self) -> str:
        function_args = self._generate_function_args()
        tensor_declarations = self._generate_tensor_declarations()
        kernel_calls = self._generate_kernel_calls()
        output_name = self.graph.outputs[0]

        return self.template.format(
            graph_name=self.graph.name,
            function_args=function_args,
            tensor_declarations=tensor_declarations,
            kernel_calls=kernel_calls,
            output_name=output_name
        )

    def _generate_function_args(self) -> str:
        args = [f"torch::Tensor {name}" for name in self.graph.inputs]
        args += [f"torch::Tensor {name}" for name in self.graph.parameters.keys()]
        return ", ".join(args)

    def _generate_tensor_declarations(self) -> str:
        declarations = []
        # Declare all tensors that are not inputs or parameters
        for name, tensor in self.graph.tensors.items():
            if name in self.graph.inputs or name in self.graph.parameters:
                continue
            shape_str = str(list(tensor.shape)).replace('[', '{').replace(']', '}')
            declarations.append(f"auto {name} = torch::empty({shape_str}, {self.graph.inputs[0]}.options());")
        return "\n    ".join(declarations)

    def _generate_kernel_calls(self) -> str:
        calls = []
        for node in self.graph.nodes:
            output = node.outputs[0]
            inputs = ", ".join(node.inputs)
            
            # THE CRITICAL CHANGE IS HERE
            if node.op_type == 'fused_gemm_relu':
                # Generate the REAL call to our custom C++ function.
                # The output tensor `relu` is the last argument.
                calls.append(f"fused_gemm_relu_forward_cuda({inputs}, {output});")
            elif node.op_type == 'matmul':
                calls.append(f"{output} = torch::matmul({inputs});")
            else:
                calls.append(f"// Unsupported op: {node.op_type}")
        return "\n    ".join(calls)

class JITCompiler:
    def __init__(self, main_cpp_source: str, kernel_sources: Dict[str, str]):
        self.main_cpp_source = main_cpp_source
        self.kernel_sources = kernel_sources
        source_hash = hashlib.md5(main_cpp_source.encode('utf-8')).hexdigest()
        self.build_dir = os.path.join(tempfile.gettempdir(), f"inferno_{source_hash}")
        os.makedirs(self.build_dir, exist_ok=True)
        self.module_name = f"inferno_module_{source_hash}"

    def compile_and_load(self):
        main_file_path = os.path.join(self.build_dir, 'main.cu')
        with open(main_file_path, 'w') as f: f.write(self.main_cpp_source)
        source_paths = [main_file_path]
        for name, content in self.kernel_sources.items():
            path = os.path.join(self.build_dir, name)
            with open(path, 'w') as f: f.write(content)
            source_paths.append(path)
        try:
            module = load(name=self.module_name, sources=source_paths, extra_ldflags=['-lcublas'], verbose=False)
            print(f"--- [JITCompiler] Compilation successful for module: {self.module_name} ---")
            return module
        except Exception as e:
            print("--- [JITCompiler] COMPILATION FAILED! ---"); raise e

class InfernoModule(torch.nn.Module):
    def __init__(self, original_model: nn.Module, compiled_kernel):
        super().__init__(); self.params = list(original_model.parameters()); self.compiled_kernel = compiled_kernel
    def forward(self, *args):
        return self.compiled_kernel.forward(*list(args), *self.params)

# ================================================================================
# SECTION 3: The Top-Level Compiler Interface
# ================================================================================

def read_kernel_file(filepath: str) -> str:
    with open(filepath, 'r') as f: return f.read()

def compile(model: nn.Module, example_inputs: List[torch.Tensor], kernel_filepath: str) -> InfernoModule:
    print("="*60); print(" " * 18 + "INFERNO COMPILER PIPELINE"); print("="*60)
    fused_kernel_code = read_kernel_file(kernel_filepath)
    parser = TorchFXParser(); ir_graph = parser.parse(model, example_inputs)
    optimizer = Optimizer(ir_graph); optimized_graph = optimizer.run_fusion_pass()
    code_gen = CodeGenerator(optimized_graph); main_cpp_source = code_gen.generate()
    jit_compiler = JITCompiler(main_cpp_source=main_cpp_source, kernel_sources={'fused_kernel.cu': fused_kernel_code})
    compiled_kernel_module = jit_compiler.compile_and_load()
    execution_engine = InfernoModule(model, compiled_kernel_module)
    print("\n--- [Inferno] Compilation pipeline complete! ---")
    return execution_engine

# ================================================================================
# DEMONSTRATION AND BENCHMARKING
# ================================================================================
def benchmark(fn, *args, warmup=10, reps=100):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps): fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps

if __name__ == '__main__':
    class MyFusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(256, 256))
            self.weight2 = torch.nn.Parameter(torch.randn(256, 256))
        def forward(self, x):
            x = torch.matmul(x, self.weight1)
            x = F.relu(x)
            y = torch.matmul(x, self.weight2)
            return y

    # --- Setup ---
    model = MyFusionModel().cuda()
    example_input = torch.randn(256, 256, device='cuda')
    KERNEL_FILE = 'src/fused_kernel.cu'

    # --- Compile the model ---
    compiled_model = compile(model, [example_input], kernel_filepath=KERNEL_FILE)

    # --- Verification ---
    print("\n--- Verifying compiled model output ---")
    original_output = model(example_input)
    compiled_output = compiled_model(example_input)
    is_close = torch.allclose(original_output, compiled_output, atol=1e-3)
    status = "✅ PASSED" if is_close else "❌ FAILED"
    print(f"Verification against original model: {status}")

    # --- Benchmarking ---
    print("\n--- Running Final Benchmark ---")
    original_latency = benchmark(model, example_input)
    compiled_latency = benchmark(compiled_model, example_input)
    speedup = original_latency / compiled_latency

    print(f"\nOriginal PyTorch Latency: {original_latency:.6f} ms")
    print(f"Inferno Compiled Latency: {compiled_latency:.6f} ms")
    print(f"🔥 Speedup: {speedup:.2f}x 🔥")

