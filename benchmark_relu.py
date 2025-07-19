import torch
import torch.nn.functional as F
import inferno_relu
import time

import torch
torch.cuda.empty_cache()

def benchmark_torch_relu(x, iters=10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        y = F.relu(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iters

def benchmark_custom_relu(x, y, iters=10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        inferno_relu.relu(x, y)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iters

def main():
    size = 1024 * 1024 * 256
    x = torch.randn(size, device='cuda')
    y = torch.empty_like(x)

    torch_time = benchmark_torch_relu(x)
    custom_time = benchmark_custom_relu(x, y)

    print(f"torch.relu:      {torch_time:.6f} ms")
    print(f"inferno_relu:    {custom_time:.6f} ms")
    print(f"speedup:         {torch_time / custom_time:.2f}x")

if __name__ == "__main__":
    main()