import os
import sys
import hashlib
import tempfile
import textwrap
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Ensure we can import from the project src/ directory (for graph and optimizer)
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from graph_creation import Graph, TorchFXParser
from graph_optimization import Optimizer


# ================================================================================
# SECTION 1: The Backend - Code Generator (from src/code_generation.py)
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
        with open(main_file_path, 'w') as f:
            f.write(self.main_cpp_source)
        source_paths = [main_file_path]
        for name, content in self.kernel_sources.items():
            path = os.path.join(self.build_dir, name)
            with open(path, 'w') as f:
                f.write(content)
            source_paths.append(path)
        try:
            module = load(name=self.module_name, sources=source_paths, extra_ldflags=['-lcublas'], verbose=False)
            print(f"--- [JITCompiler] Compilation successful for module: {self.module_name} ---")
            return module
        except Exception as e:
            print("--- [JITCompiler] COMPILATION FAILED! ---")
            raise e


class InfernoModule(torch.nn.Module):
    def __init__(self, original_model: nn.Module, compiled_kernel):
        super().__init__()
        self.params = list(original_model.parameters())
        self.compiled_kernel = compiled_kernel

    def forward(self, *args):
        return self.compiled_kernel.forward(*list(args), *self.params)


# ================================================================================
# SECTION 2: Top-Level Compiler Interface (from src/code_generation.py)
# ================================================================================
def _read_kernel_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()


def compile(model: nn.Module, example_inputs: List[torch.Tensor], kernel_filepath: str) -> InfernoModule:
    print("="*60)
    print(" " * 18 + "INFERNO COMPILER PIPELINE")
    print("="*60)
    fused_kernel_code = _read_kernel_file(kernel_filepath)
    parser = TorchFXParser()
    ir_graph = parser.parse(model, example_inputs)
    optimizer = Optimizer(ir_graph)
    optimized_graph = optimizer.run_fusion_pass()
    code_gen = CodeGenerator(optimized_graph)
    main_cpp_source = code_gen.generate()
    jit_compiler = JITCompiler(main_cpp_source=main_cpp_source, kernel_sources={'fused_kernel.cu': fused_kernel_code})
    compiled_kernel_module = jit_compiler.compile_and_load()
    execution_engine = InfernoModule(model, compiled_kernel_module)
    print("\n--- [Inferno] Compilation pipeline complete! ---")
    return execution_engine


# ================================================================================
# SECTION 3: Public Decorator API for PyTorch Models
# ================================================================================
_DEFAULT_KERNEL_FILE = os.path.join(_SRC_DIR, "fused_kernel.cu")


def compile_model(example_inputs: List[torch.Tensor], kernel_filepath: str = _DEFAULT_KERNEL_FILE):
    """
    Class decorator to compile a PyTorch nn.Module using Inferno at instantiation time.

    Usage:
        @compile_model(example_inputs=[torch.randn(256, 256, device='cuda')])
        class MyModel(nn.Module):
            ...

        model = MyModel().cuda()
        out = model(x)  # Uses compiled fused kernel

    Notes:
    - The provided `example_inputs` should match the inputs you'll pass to `forward`.
    - Compilation happens in the decorated class __init__, after original initialization.
    """
    def _decorator(ModelCls):
        class InfernoCompiledModel(ModelCls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Perform compilation using the fully constructed model
                self._inferno_compiled = compile(self, example_inputs, kernel_filepath)

            def forward(self, *args, **kwargs):
                # Delegate forward to the compiled execution engine
                return self._inferno_compiled(*args, **kwargs)

        InfernoCompiledModel.__name__ = ModelCls.__name__
        InfernoCompiledModel.__qualname__ = ModelCls.__qualname__
        return InfernoCompiledModel

    return _decorator