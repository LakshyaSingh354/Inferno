import torch
import inferno_fused

def benchmark_fused_gemm_relu(M, K, N):
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warm-up (important for stable profiling)
    for _ in range(5):
        _ = inferno_fused.fused_gemm_relu(A, B)
    
    # Actual run (this is what ncu will capture)
    C = inferno_fused.fused_gemm_relu(A, B)
    torch.cuda.synchronize()  # make sure kernel finishes
    return C

if __name__ == "__main__":
    M = N = K = 1024
    print(f"Running fused GEMM+ReLU for {M}x{K} * {K}x{N}")
    out = benchmark_fused_gemm_relu(M, K, N)
    print("Done.")