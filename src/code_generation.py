import hashlib
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import List, Dict, Any
import textwrap
from src.graph_creation import Graph, TorchFXParser
from src.graph_optimization import Optimizer

# ================================================================================
# SECTION 1: The Backend - Code Generator
# ================================================================================

class CodeGenerator:
    """
    Takes an optimized Inferno IR Graph and generates compilable C++/CUDA source code.
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        # This template now includes a global cuBLAS handle, essential for performance.
        self.template = textwrap.dedent("""\
            #include <torch/extension.h>
            #include <cublas_v2.h>
            #include <vector>

            // Forward declaration of our actual fused kernel implementation
            void fused_gemm_relu_forward_cuda(cublasHandle_t, torch::Tensor, torch::Tensor, torch::Tensor);

            // Global cuBLAS handle
            cublasHandle_t get_cublas_handle() {{
                static bool initialized = false;
                static cublasHandle_t handle;
                if (!initialized) {{
                    cublasCreate(&handle);
                    initialized = true;
                }}
                return handle;
            }}

            // The main forward function for the compiled model
            torch::Tensor {graph_name}_forward(
                {function_args}
            ) {{
                // Get the cuBLAS handle
                cublasHandle_t handle = get_cublas_handle();

                // Intermediate tensor declarations
                {tensor_declarations}

                // Sequence of kernel calls
                {kernel_calls}

                // Return the final output tensor
                return {output_name};
            }}

            // Pybind11 wrapper to expose the forward function to Python
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
        args += [f"torch::Tensor {name}" for name in self.graph.parameters]
        return ", ".join(args)

    def _generate_tensor_declarations(self) -> str:
        declarations = []
        for name, tensor in self.graph.tensors.items():
            if name in self.graph.inputs or name in self.graph.parameters:
                continue
            shape_str = str(list(tensor.shape)).replace('[', '{').replace(']', '}')
            # Use .options() to ensure tensors are created on the correct device (e.g., 'cuda')
            declarations.append(f"auto {name} = torch::empty({shape_str}, {self.graph.inputs[0]}.options());")
        return "\n    ".join(declarations)

    def _generate_kernel_calls(self) -> str:
        calls = []
        for node in self.graph.nodes:
            output = node.outputs[0]
            inputs = ", ".join(node.inputs)
            if node.op_type == 'fused_gemm_relu':
                # This now generates the REAL call to our custom C++ function.
                calls.append(f"fused_gemm_relu_forward_cuda(handle, {inputs}, {output});")
            elif node.op_type == 'matmul':
                calls.append(f"{output} = torch::matmul({inputs});")
            else:
                calls.append(f"// Unsupported op: {node.op_type}")
        return "\n    ".join(calls)

class JITCompiler:
    """
    Takes C++ source code, writes it to files, compiles it into a shared library
    using PyTorch's JIT compiler, and loads it into the current process.
    """
    def __init__(self, main_cpp_source: str, kernel_sources: Dict[str, str]):
        self.main_cpp_source = main_cpp_source
        self.kernel_sources = kernel_sources # e.g., {'fused_kernel.cu': '...source...'}
        
        # Create a unique build directory for each compilation
        source_hash = hashlib.md5(main_cpp_source.encode('utf-8')).hexdigest()
        self.build_dir = os.path.join(tempfile.gettempdir(), f"inferno_{source_hash}")
        os.makedirs(self.build_dir, exist_ok=True)
        
        self.module_name = f"inferno_module_{source_hash}"

    def compile_and_load(self):
        print(f"--- [JITCompiler] Compiling module: {self.module_name} ---")
        
        # Write source files to the temporary build directory
        main_file_path = os.path.join(self.build_dir, 'main.cpp')
        with open(main_file_path, 'w') as f:
            f.write(self.main_cpp_source)
            
        source_paths = [main_file_path]
        for name, content in self.kernel_sources.items():
            path = os.path.join(self.build_dir, name)
            with open(path, 'w') as f:
                f.write(content)
            source_paths.append(path)

        try:
            # Use PyTorch's more powerful `load` function
            module = load(
                name=self.module_name,
                sources=source_paths,
                extra_ldflags=['-lcublas'], # Crucially, we link against the cuBLAS library
                verbose=False # Set to True to see nvcc/g++ commands
            )
            print("--- [JITCompiler] Compilation successful. ---")
            return module
        except Exception as e:
            print("--- [JITCompiler] COMPILATION FAILED! ---")
            raise e

class InfernoModule(torch.nn.Module):
    """
    An execution engine. This nn.Module holds the compiled kernel and the
    model's parameters, wiring them together for execution.
    """
    def __init__(self, original_model: nn.Module, compiled_kernel):
        super().__init__()
        self.params = list(original_model.parameters())
        self.compiled_kernel = compiled_kernel

    def forward(self, *args):
        full_args = list(args) + self.params
        return self.compiled_kernel.forward(*full_args)

# ================================================================================
# SECTION 3: The Top-Level Compiler Interface
# ================================================================================

# We need the source code of our actual fused kernel from Milestone 2
FUSED_KERNEL_CODE = textwrap.dedent("""\
    #include <torch/extension.h>
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    
    __global__ void relu_kernel(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(data[idx], 0.0f);
        }
    }

    void fused_gemm_relu_forward_cuda(
        cublasHandle_t handle,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor C) {

        int M = A.size(0);
        int K = A.size(1);
        int N = B.size(1);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta,
                    C.data_ptr<float>(), N);

        const int total_elements = M * N;
        const int threads_per_block = 256;
        const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        relu_kernel<<<blocks_per_grid, threads_per_block>>>(C.data_ptr<float>(), total_elements);
    }
""")

def compile(model: nn.Module, example_inputs: List[torch.Tensor]) -> InfernoModule:
    """
    The main end-to-end compiler function.
    """
    print("="*60)
    print(" " * 18 + "INFERNO COMPILER PIPELINE")
    print("="*60)
    
    parser = TorchFXParser()
    ir_graph = parser.parse(model, example_inputs)
    
    optimizer = Optimizer(ir_graph)
    optimized_graph = optimizer.run_fusion_pass()
    
    code_gen = CodeGenerator(optimized_graph)
    main_cpp_source = code_gen.generate()
    
    # We now pass all necessary source files to the JIT compiler
    jit_compiler = JITCompiler(
        main_cpp_source=main_cpp_source,
        kernel_sources={'fused_kernel.cu': FUSED_KERNEL_CODE}
    )
    compiled_kernel_module = jit_compiler.compile_and_load()

    execution_engine = InfernoModule(model, compiled_kernel_module)
    
    print("\n--- [Inferno] Compilation pipeline complete! ---")
    return execution_engine


# ================================================================================
# DEMONSTRATION
# ================================================================================
if __name__ == '__main__':
    class MyFusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(512, 128))
            self.weight2 = torch.nn.Parameter(torch.randn(128, 64))
        def forward(self, x):
            x = torch.matmul(x, self.weight1)
            x = F.relu(x)
            y = torch.matmul(x, self.weight2)
            return y

    model = MyFusionModel().cuda()
    example_input = torch.randn(256, 512, device='cuda')
    
    compiled_model = compile(model, [example_input])

    print("\n--- Verifying compiled model output ---")
    original_output = model(example_input)
    compiled_output = compiled_model(example_input)

    is_close = torch.allclose(original_output, compiled_output, atol=1e-5)
    status = "✅ PASSED" if is_close else "❌ FAILED"
    print(f"Verification against original model: {status}")
    if not is_close:
        print("Max difference:", (original_output - compiled_output).abs().max().item())

