from graph_creation import Graph, OperatorNode, TorchFXParser
import torch
import torch.nn.functional as F

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