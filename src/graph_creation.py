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

class TorchFXParser:
    def __init__(self):
        self.op_map = {
            'torch.matmul': 'matmul',
            'torch.relu': 'relu',
            # Add other op mappings here as we support them
        }

    def parse(self, model: torch.nn.Module, example_inputs: List[torch.Tensor]) -> Graph:
        print(f"--- [Parser] Parsing model: {model.__class__.__name__} ---")
        
        # 1. Use symbolic_trace to get the FX graph
        traced_model = fx.symbolic_trace(model)
        
        # Run shape propagation to fill in tensor_meta
        fx.passes.shape_prop.ShapeProp(traced_model).propagate(*example_inputs)
        
        # Create a new, empty Inferno graph
        inferno_graph = Graph(name=model.__class__.__name__)

        param_meta_map = {name: p.shape for name, p in model.named_parameters()}

        # 2. Iterate through the nodes of the FX graph
        for fx_node in traced_model.graph.nodes:
            if fx_node.op == 'placeholder':
                # This is a graph input
                tensor_meta = fx_node.meta['tensor_meta']
                tensor_node = TensorNode(fx_node.name, tuple(tensor_meta.shape), str(tensor_meta.dtype))
                inferno_graph.add_tensor(tensor_node)
                inferno_graph.inputs.append(fx_node.name)

            elif fx_node.op == 'get_attr':
                # This is a model parameter (e.g., self.weight)
                # Let's get its metadata from the original model
                shape = tuple(param_meta_map[fx_node.target])
                # We assume parameters are float32 for now
                tensor_node = TensorNode(fx_node.name, shape, "torch.float32") 
                inferno_graph.add_tensor(tensor_node)
                inferno_graph.add_parameter(tensor_node)

            elif fx_node.op == 'call_function':
                # This is an operation
                op_type = self.op_map.get(str(fx_node.target), str(fx_node.target))
                
                # Get input tensor names
                input_names = [str(arg.name) for arg in fx_node.args]
                
                # Create the output tensor node
                tensor_meta = fx_node.meta['tensor_meta']
                output_tensor = TensorNode(fx_node.name, tuple(tensor_meta.shape), str(tensor_meta.dtype))
                inferno_graph.add_tensor(output_tensor)
                
                # Create the operator node
                op_node = OperatorNode(
                    name=fx_node.name,
                    op_type=op_type,
                    inputs=input_names,
                    outputs=[fx_node.name]
                )
                inferno_graph.add_node(op_node)

            elif fx_node.op == 'output':
                # This defines the graph's final output(s)
                # fx_node.args[0] might be a single node or a tuple of nodes
                output_arg = fx_node.args[0]
                if isinstance(output_arg, (tuple, list)):
                    output_names = [str(arg.name) for arg in output_arg]
                else:
                    output_names = [str(output_arg.name)]
                inferno_graph.outputs.extend(output_names)

        print("--- [Parser] Parsing complete. ---")
        return inferno_graph


# ================================================================================
# DEMONSTRATION
# ================================================================================
if __name__ == '__main__':
    # Define a simple PyTorch model
    class MySimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # We can define weights here, but for tracing they are treated as inputs
            self.weight = torch.nn.Parameter(torch.randn(512, 128))

        def forward(self, x):
            # The pattern we want our compiler to understand
            x = torch.matmul(x, self.weight)
            x = torch.relu(x)

            return x

    # --- Run the Parser ---
    model = MySimpleModel()
    parser = TorchFXParser()

    # We need example inputs to allow FX to trace the model and infer shapes/dtypes
    example_input = torch.randn(256, 512)

    # PyTorch model -> Inferno IR
    # Note: FX treats model parameters like `self.weight` as inputs to the graph
    ir_graph = parser.parse(model, [example_input])

    # --- Print the resulting IR ---
    ir_graph.print()
