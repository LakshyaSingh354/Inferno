import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from code_generation import compile

# ================================================================================
# Minimal script to profile the compiled Inferno Module execution.
# Purpose: Isolate the compiled model's forward pass for ncu.
# ================================================================================

# --- Configuration ---
M, K, N = 64, 64, 64
REPS = 100
KERNEL_FILE = 'src/fused_kernel.cu'

def main():
    """
    Compiles the model once, then runs the compiled version in a loop for profiling.
    """
    print(f"--- Profiling Inferno Compiled: MatMul({M},{K}) + ReLU({K},{N}) ---")

    # --- 1. Setup and One-Time Compilation (NOT PROFILED) ---
    class ModelToCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(K, N))
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x

    model = ModelToCompile().cuda()
    example_input = torch.randn(M, K, device='cuda')

    print("    Compiling model (this part is not profiled)...")
    compiled_model = compile(model, [example_input], kernel_filepath=KERNEL_FILE)

    # --- 2. The Profiling Loop ---
    # ncu will attach to this part.
    for _ in range(REPS):
        # This is the single, fast call to our compiled code.
        output = compiled_model(example_input)

    # Ensure all GPU work is finished
    torch.cuda.synchronize()
    print("--- Inferno Compiled run complete. ---")

if __name__ == "__main__":
    main()
