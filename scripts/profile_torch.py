import torch
import torch.nn.functional as F

# ================================================================================
# Minimal script to profile the baseline PyTorch eager mode execution.
# Purpose: Isolate the vanilla torch.relu(torch.matmul(...)) operations for ncu.
# ================================================================================

# --- Configuration ---
M, K, N = 64, 64, 64
REPS = 100

def main():
    """
    Sets up tensors and runs the target operations in a loop for profiling.
    """
    print(f"--- Profiling PyTorch Eager: MatMul({M},{K}) + ReLU({K},{N}) ---")

    # Create input tensors on the GPU
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # --- The Profiling Loop ---
    # ncu will attach to this part. We run it multiple times to ensure
    # the profiler captures a stable workload.
    for _ in range(REPS):
        # This is the exact operation sequence we want to analyze.
        C = F.relu(torch.matmul(A, B))

    # Ensure all GPU work is finished before the script exits
    torch.cuda.synchronize()
    print("--- PyTorch Eager run complete. ---")


if __name__ == "__main__":
    main()
