from src.graph_creation import Graph, OperatorNode, TorchFXParser
import torch
import torch.nn.functional as F
from scripts.graph_visualize import Visualizer

class Optimizer:
    """
    Container for optimization passes that rewrite the Inferno IR Graph.
    """
    def __init__(self, graph: Graph):
        self.graph = graph

    def run_fusion_pass(self):
        """
        Finds and fuses `matmul` -> `relu` patterns in the graph.
        This is a simple, greedy fusion pass that modifies the graph in-place.
        """
        print("\n--- [Optimizer] Running MatMul+ReLU Fusion Pass ---")
        
        # We iterate backwards to make node removal safer
        # (removing a node doesn't affect the indices of nodes we've already processed)
        for i in range(len(self.graph.nodes) - 1, 0, -1):
            current_node = self.graph.nodes[i]
            prev_node = self.graph.nodes[i-1]

            # PATTERN MATCHING: A `relu` node immediately following a `matmul` node
            # AND the matmul's output is the relu's only input.
            is_relu = current_node.op_type == 'relu'
            is_matmul = prev_node.op_type == 'matmul'
            is_connected = len(current_node.inputs) == 1 and current_node.inputs[0] == prev_node.outputs[0]

            if is_relu and is_matmul and is_connected:
                # FUSION! We found the pattern.
                print(f"    🔥 Fusing '{prev_node.name}' and '{current_node.name}'")
                
                # Create the new fused node.
                # It takes the inputs of the matmul and produces the output of the relu.
                fused_node = OperatorNode(
                    name=f"fused_{prev_node.name}_{current_node.name}",
                    op_type='fused_gemm_relu',
                    inputs=prev_node.inputs,
                    outputs=current_node.outputs
                )
                
                # --- In-place Graph Rewrite ---
                # 1. Replace the `relu` node with our new `fused_node`.
                self.graph.nodes[i] = fused_node
                
                # 2. Remove the old `matmul` node.
                del self.graph.nodes[i-1]

                # Note: We don't need to explicitly "re-wire" subsequent nodes
                # because the output tensor name (`relu`) remains the same.
                # The next node that needs the 'relu' tensor will find it
                # as the output of our new fused node.

        print("--- [Optimizer] Fusion Pass Complete ---")
        return self.graph


# ================================================================================
# DEMONSTRATION
# ================================================================================
if __name__ == '__main__':
    # Define a model with the pattern we want to fuse
    class MyFusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(512, 128))
            self.weight2 = torch.nn.Parameter(torch.randn(128, 64))

        def forward(self, x):
            x = torch.matmul(x, self.weight1) # This matmul...
            x = F.relu(x)                     # ...and this relu should be fused.
            
            x = torch.matmul(x, self.weight2) # This matmul should NOT be fused.
            y = torch.softmax(x, dim=1)
            return y

    # --- Run the full pipeline: Parse -> Optimize -> Visualize ---
    
    # 1. Parse the model into our IR
    model = MyFusionModel()
    parser = TorchFXParser()
    example_input = torch.randn(256, 512)
    ir_graph = parser.parse(model, [example_input])

    # 2. Visualize the ORIGINAL graph
    print("\n--- VISUALIZING ORIGINAL GRAPH ---")
    viz_original = Visualizer(ir_graph)
    viz_original.visualize('original_graph')

    # 3. Run the optimization pass
    optimizer = Optimizer(ir_graph)
    optimized_graph = optimizer.run_fusion_pass()
    optimized_graph.print()

    # 4. Visualize the OPTIMIZED graph
    print("\n--- VISUALIZING OPTIMIZED GRAPH ---")
    viz_optimized = Visualizer(optimized_graph)
    viz_optimized.visualize('optimized_graph')
