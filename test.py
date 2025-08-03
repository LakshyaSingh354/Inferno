import torch
import inferno_fused

def check(label, a, b):
    """Helper function to print comparison results."""
    is_close = torch.allclose(a, b, atol=1e-5)
    status = "✅ PASSED" if is_close else "❌ FAILED"
    print(f"{label:<40} {status}")
    if not is_close:
        # Print the difference for more detail on failure
        diff = torch.abs(a - b).max()
        print(f"    └─ Max difference: {diff.item()}")

def main():
    print("="*60)
    print(" " * 15 + "INFERNO COMPILER DEBUG HARNESS")
    print("="*60)

    # --- Setup ---
    M, K, N = 256, 512, 128
    print(f"Using matrix sizes: A({M}, {K}), B({K}, {N})\n")
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    # An intermediate tensor for testing ReLU
    X = torch.randn(M, N, device='cuda')

    # --- Forensic Analysis ---
    
    # Test 1: Isolate the Matrix Multiplication
    print("--- STAGE 1: Analyzing GEMM Component ---")
    pytorch_matmul = torch.matmul(A, B)
    inferno_matmul = inferno_fused.debug_gemm_only(A, B)
    check("Comparing PyTorch MatMul vs. Inferno GEMM", pytorch_matmul, inferno_matmul)

    # Test 2: Isolate the ReLU Kernel
    print("\n--- STAGE 2: Analyzing ReLU Component ---")
    pytorch_relu = torch.relu(X)
    inferno_relu = inferno_fused.debug_relu_only(X)
    check("Comparing PyTorch ReLU vs. Inferno ReLU", pytorch_relu, inferno_relu)

    # Test 3: Test the final, corrected fused operation
    print("\n--- STAGE 3: Analyzing Corrected Fused Op ---")
    pytorch_fused = torch.relu(torch.matmul(A, B))
    inferno_fused_corrected = inferno_fused.fused_gemm_relu(A, B)
    check("Comparing PyTorch vs. Corrected Fused Op", pytorch_fused, inferno_fused_corrected)
    
    print("\n" + "="*60)
    print("Analysis complete.")


if __name__ == "__main__":
    try:
        _ = inferno_fused.debug_gemm_only
    except AttributeError:
        print("ERROR: Debug functions not found. Did you recompile with the new C++ code?")
        exit()
    main()
