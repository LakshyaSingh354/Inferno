import torch
import torch.fx as fx
from typing import List, Dict, Any
import torch.nn.functional as F


# ================================================================================
# SECTION 1: The Intermediate Representation (IR) Definition
# ================================================================================

class TensorNode:
    """Represents a tensor in the computation graph (an edge)."""
    def __init__(self, name: str, shape: tuple, dtype: str):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"Tensor('{self.name}', shape={self.shape}, dtype={self.dtype})"

class OperatorNode:
    """Represents an operation in the computation graph (a node)."""
    def __init__(self, name: str, op_type: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any] = None):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs    # List of names of input TensorNodes
        self.outputs = outputs  # List of names of output TensorNodes
        self.attrs = attrs if attrs is not None else {}

    def __repr__(self):
        return f"Op('{self.name}', type='{self.op_type}', inputs={self.inputs}, outputs={self.outputs})"

class Graph:
    """The IR graph, containing all nodes and tensors."""
    def __init__(self, name: str):
        self.name = name
        self.nodes: List[OperatorNode] = []
        self.tensors: Dict[str, TensorNode] = {}
        self.parameters: Dict[str, TensorNode] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def add_node(self, node: OperatorNode):
        self.nodes.append(node)

    def add_tensor(self, tensor: TensorNode):
        self.tensors[tensor.name] = tensor

    def add_parameter(self, parameter: TensorNode):
        self.parameters[parameter.name] = parameter

    def print(self):
        """Prints a readable representation of the graph."""
        print(f"--- Inferno IR: Graph('{self.name}') ---")
        print("Inputs:")
        for name in self.inputs:
            print(f"  - {self.tensors[name]}")

        print("\nParameters:")
        for name in self.parameters:
            print(f"  - {self.parameters[name]}")
        
        print("\nNodes:")
        for node in self.nodes:
            print(f"  - {node}")

        print("\nOutputs:")
        for name in self.outputs:
            print(f"  - {self.tensors[name]}")
        print("------------------------------------------")


# ================================================================================
# SECTION 2: The Torch FX to Inferno IR Parser
# ================================================================================
# This class is responsible for taking a PyTorch model traced by FX and
# converting it into our custom Graph IR.

class FlatteningTracer(fx.Tracer):
    """
    A custom FX Tracer that does not treat nn.Module instances as leaf modules.
    This allows us to trace *into* layers like nn.Linear and see the underlying
    matmul and add operations.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualname: str) -> bool:
        # If we want to keep some modules as black boxes, we can add them here.
        # For now, we want to trace into everything, so we always return False.
        return False

class TorchFXParser:
    def __init__(self):
        # Let's make our op_map more robust by using the functions themselves as keys
        self.op_map = {
            torch.matmul: 'matmul',
            F.relu: 'relu',
            F.normalize: 'normalize',
            torch.add: 'add',
            F.softmax: 'softmax',
            # Note: The matmul inside nn.Linear is actually torch.addmm,
            # but FX often decomposes it. We'll handle what FX gives us.
        }

    def parse(self, model: torch.nn.Module, example_inputs: List[torch.Tensor]) -> Graph:
        print(f"--- [Parser] Parsing model: {model.__class__.__name__} ---")
        
        # 1. Use our custom FlatteningTracer
        tracer = FlatteningTracer()
        graph = tracer.trace(model)
        traced_model = fx.GraphModule(tracer.root, graph)
        
        # Run shape propagation
        fx.passes.shape_prop.ShapeProp(traced_model).propagate(*example_inputs)
        
        inferno_graph = Graph(name=model.__class__.__name__)
        param_meta_map = {name: p.shape for name, p in model.named_parameters()}

        for fx_node in traced_model.graph.nodes:
            if fx_node.op == 'placeholder':
                tensor_meta = fx_node.meta['tensor_meta']
                tensor_node = TensorNode(fx_node.name, tuple(tensor_meta.shape), str(tensor_meta.dtype))
                inferno_graph.add_tensor(tensor_node)
                inferno_graph.inputs.append(fx_node.name)

            elif fx_node.op == 'get_attr':
                shape = tuple(param_meta_map[fx_node.target])
                tensor_node = TensorNode(fx_node.name, shape, "torch.float32")
                inferno_graph.add_tensor(tensor_node)
                inferno_graph.add_parameter(tensor_node)

            elif fx_node.op == 'call_function':
                # Use the function object directly for lookup
                op_type = self.op_map.get(fx_node.target, "unknown_op")
                
                # This needs to handle args and kwargs more robustly
                input_names = []
                for arg in fx_node.args:
                    if isinstance(arg, fx.Node):
                        input_names.append(arg.name)
                # We'll ignore kwargs for now for simplicity
                
                tensor_meta = fx_node.meta['tensor_meta']
                output_tensor = TensorNode(fx_node.name, tuple(tensor_meta.shape), str(tensor_meta.dtype))
                inferno_graph.add_tensor(output_tensor)
                
                op_node = OperatorNode(
                    name=fx_node.name,
                    op_type=op_type,
                    inputs=input_names,
                    outputs=[fx_node.name]
                )
                inferno_graph.add_node(op_node)

            elif fx_node.op == 'output':
                output_arg = fx_node.args[0]
                if isinstance(output_arg, (tuple, list)):
                    output_names = [str(arg.name) for arg in output_arg]
                else:
                    output_names = [str(output_arg.name)]
                inferno_graph.outputs.extend(output_names)

        print("--- [Parser] Parsing complete. ---")
        return inferno_graph
