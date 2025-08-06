import torch

import inferno

torch.cuda.init()
import torch.nn.functional as F

@inferno.kernel('relu_vec4')
def fast_relu(x, y):
    return torch.relu(x, y)

def benchmark_torch_relu(x, iters=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    for _ in range(iters):
        y = F.relu(x)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms


def benchmark_custom_relu(x, y, iters=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    for _ in range(iters):
        fast_relu(x, y)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms


def main():
    size = 1024 * 1024 * 4
    x = torch.randn(size, device='cuda')
    y = torch.empty_like(x)

    torch_time = benchmark_torch_relu(x)
    custom_time = benchmark_custom_relu(x, y)

    print(f"torch.relu:      {torch_time:.6f} ms")
    print(f"inferno_relu:    {custom_time:.6f} ms")
    print(f"speedup:         {torch_time / custom_time:.2f}x")

if __name__ == "__main__":
    main()