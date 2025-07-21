import torch
import time
import inferno_fused

torch.manual_seed(0)

def benchmark(func, warmup=5, iters=100):
    # Warmup
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        func()
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) * 1000 / iters  # ms per iteration

# Shapes: A [m x k], B [k x n] -> C [m x n]
m, k, n = 2048, 2048, 2048
A = torch.randn(m, k, device='cuda', dtype=torch.float32)
B = torch.randn(k, n, device='cuda', dtype=torch.float32)

# Result tensors
C_torch = torch.empty(m, n, device='cuda')
C_inferno = torch.empty_like(C_torch)

# Torch version
def torch_baseline():
    C_torch = torch.relu(torch.matmul(A, B))

# Inferno fused version
def inferno_fused_kernel():
    inferno_fused.matmul_relu(A, B, C_inferno)

# ✅ Correctness check
expected = torch.relu(A @ B)
inferno_fused.matmul_relu(A, B, C_inferno)
assert torch.allclose(C_inferno, expected, atol=1e-4), "Mismatch between fused and baseline!"

# ⏱️ Benchmark both
torch_time = benchmark(torch_baseline)
inferno_time = benchmark(inferno_fused_kernel)

# 📊 Print results
print(f"torch.relu(torch.matmul): {torch_time:.5f} ms")
print(f"inferno fused kernel:     {inferno_time:.5f} ms")
print(f"Speedup:                  {torch_time / inferno_time:.2f}x")