from src.graph_creation import Graph, TorchFXParser
from graphviz import Digraph
import torch
import torch.nn.functional as F

class Visualizer:
    """
    Renders an Inferno IR Graph into a beautiful diagram using Graphviz.
    """
    def __init__(self, graph: Graph):
        self.graph = graph
        self.dot = Digraph(name=graph.name, comment='Inferno Compiler IR Graph')
        
        # Define styles for different node types
        self.styles = {
            'input': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#a6e3a1', 'fontname': 'Helvetica'},
            'output': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#89b4fa', 'fontname': 'Helvetica'},
            'operator': {'shape': 'box', 'style': 'filled', 'fillcolor': '#f9e2af', 'fontname': 'Helvetica'},
            'parameter': {'shape': 'box3d', 'style': 'filled', 'fillcolor': '#cdd6f4', 'fontname': 'Helvetica'},
        }

    def visualize(self, filename='inferno_graph.gv'):
        """
        Builds and renders the Graphviz graph.
        """
        print(f"\n--- [Visualizer] Rendering graph to {filename}.png ---")
        self.dot.attr(rankdir='TB', splines='ortho', bgcolor='transparent')

        # Add input and parameter nodes
        with self.dot.subgraph(name='cluster_inputs') as c:
            c.attr(label='Inputs & Parameters', style='filled', color='#313244', fontcolor='white')
            for name in self.graph.inputs:
                tensor = self.graph.tensors[name]
                label = f"{tensor.name}\n{tensor.shape}\n{str(tensor.dtype).replace('torch.', '')}"
                c.node(name, label, **self.styles['input'])
            for name, tensor in self.graph.parameters.items():
                label = f"{tensor.name}\n{tensor.shape}"
                c.node(name, label, **self.styles['parameter'])
        
        # Add operator nodes
        for op_node in self.graph.nodes:
            label = f"{op_node.name}\n({op_node.op_type})"
            self.dot.node(op_node.name, label, **self.styles['operator'])

        # Add output nodes
        with self.dot.subgraph(name='cluster_outputs') as c:
            c.attr(label='Outputs', style='filled', color='#313244', fontcolor='white')
            for name in self.graph.outputs:
                tensor = self.graph.tensors[name]
                label = f"{tensor.name}\n{tensor.shape}\n{str(tensor.dtype).replace('torch.', '')}"
                c.node(name, label, **self.styles['output'])

        # Add edges to connect the graph
        for op_node in self.graph.nodes:
            for input_name in op_node.inputs:
                self.dot.edge(input_name, op_node.name)
            for output_name in op_node.outputs:
                # Find the next operator that uses this output
                for next_op in self.graph.nodes:
                    if output_name in next_op.inputs:
                        self.dot.edge(output_name, next_op.name)
                # If this is a final graph output, connect it to the output node
                if output_name in self.graph.outputs:
                    self.dot.edge(output_name, output_name)

        # Render the graph
        try:
            self.dot.render(filename, format='png', view=False, cleanup=True)
            print(f"--- [Visualizer] Success! Graph saved to {filename}.png ---")
        except Exception as e:
            print("\n" + "="*80)
            print("  GRAPHVIZ ERROR: Could not render the graph.")
            print("  Please ensure you have installed the Graphviz system package.")
            print("  - On Ubuntu/Debian: sudo apt-get install graphviz")
            print("  - On macOS (with Homebrew): brew install graphviz")
            print("  - On Windows: Download from https://graphviz.org/download/ and add to PATH.")
            print(f"  Python library error: {e}")
            print("="*80)


# ================================================================================
# DEMONSTRATION
# ================================================================================
if __name__ == '__main__':
    # Define a more complex PyTorch model to show off the visualizer
    class MyComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(512, 128))
            self.linear1 = torch.nn.Linear(128, 64)
            self.linear2 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.softmax(x, dim=1)
            return x

    # --- Run the Parser ---
    model = MyComplexModel()
    parser = TorchFXParser()
    example_input = torch.randn(256, 512)
    ir_graph = parser.parse(model, [example_input])
    ir_graph.print()
    
    visualizer = Visualizer(ir_graph)
    visualizer.visualize('results/complex_model_graph')