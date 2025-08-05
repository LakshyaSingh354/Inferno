import torch
import time
from inferno import InfernoCompiler

# ================================================================================
# DEMONSTRATION
# ================================================================================

inferno = InfernoCompiler()

# --- Define a model using the pattern ---
@inferno.compile
def simple_fused_model(A, B):
    # This code will be analyzed by the decorator.
    # If the pattern is found, this body will be replaced.
    return torch.relu(torch.matmul(A, B))

# --- Define another model WITHOUT the pattern ---
@inferno.compile
def simple_unfused_model(A, B):
    x = torch.matmul(A, B)
    # Some other operation in between
    x = x * 2.0 
    y = torch.relu(x)
    return y

# --- Benchmarking utilities ---
def benchmark_function(func, A, B, num_runs=100, warmup_runs=10):
    """Benchmark a function with warmup runs and timing statistics."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(A, B)
    
    # Synchronize GPU before timing
    if A.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing runs
    start_time = time.time()
    for _ in range(num_runs):
        _ = func(A, B)
    
    # Synchronize GPU after timing
    if A.device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = num_runs / total_time
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': throughput,
        'num_runs': num_runs
    }

def print_benchmark_results(name, results):
    """Print formatted benchmark results."""
    print(f"\n    {name}:")
    print(f"      Average time: {results['avg_time']*1000:.3f} ms")
    print(f"      Total time: {results['total_time']:.3f} s")
    print(f"      Throughput: {results['throughput']:.1f} ops/sec")


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" " * 10 + "INFERNO COMPILER DECORATOR DEMO")
    print("="*60)

    # --- Setup ---
    A = torch.randn(256, 512, device='cuda')
    B = torch.randn(512, 128, device='cuda')

    # --- Test the compiled function ---
    print("\n[1] Testing the model that SHOULD be fused...")
    
    # When we call this, we're actually calling the `compiled_wrapper`
    # that our decorator returned.
    output_fused = simple_fused_model(A, B)
    
    # For verification, let's call the PyTorch version directly
    output_pytorch = torch.relu(torch.matmul(A, B))
    
    is_close = torch.allclose(output_fused, output_pytorch, atol=1e-5)
    status = "✅ PASSED" if is_close else "❌ FAILED"
    print(f"\n    Verification against PyTorch: {status}")
    print(f"    Output shape: {output_fused.shape}")

    # --- Test the un-compiled function ---
    print("\n[2] Testing the model that should NOT be fused...")
    
    # This should have returned the original function
    output_unfused = simple_unfused_model(A, B)
    
    # Verification
    output_pytorch_unfused = torch.relu(torch.matmul(A, B) * 2.0)
    
    is_close_unfused = torch.allclose(output_unfused, output_pytorch_unfused, atol=1e-5)
    status_unfused = "✅ PASSED" if is_close_unfused else "❌ FAILED"
    print(f"\n    Verification against PyTorch: {status_unfused}")
    print(f"    Output shape: {output_unfused.shape}")
    
    # --- Performance Benchmarking ---
    print("\n[3] PERFORMANCE BENCHMARKING")
    print("-" * 40)
    
    # Define PyTorch baseline functions for comparison
    def pytorch_fused_baseline(A, B):
        return torch.relu(torch.matmul(A, B))
    
    def pytorch_unfused_baseline(A, B):
        x = torch.matmul(A, B)
        y = torch.relu(x)
        return y
    
    # Benchmark parameters
    num_runs = 1000
    warmup_runs = 50
    
    print(f"\nBenchmarking with {num_runs} runs (after {warmup_runs} warmup runs):")
    print(f"Matrix sizes: A({A.shape}) @ B({B.shape})")
    
    # Benchmark fused model (should use custom kernel)
    print("\n--- FUSED MODEL BENCHMARKS ---")
    fused_results = benchmark_function(simple_fused_model, A, B, num_runs, warmup_runs)
    print_benchmark_results("Inferno Fused", fused_results)
    
    # Benchmark PyTorch fused baseline
    pytorch_fused_results = benchmark_function(pytorch_fused_baseline, A, B, num_runs, warmup_runs)
    print_benchmark_results("PyTorch Fused", pytorch_fused_results)
    
    # Benchmark unfused model (should use original PyTorch)
    print("\n--- UNFUSED MODEL BENCHMARKS ---")
    unfused_results = benchmark_function(simple_unfused_model, A, B, num_runs, warmup_runs)
    print_benchmark_results("Inferno Unfused", unfused_results)
    
    # Benchmark PyTorch unfused baseline
    pytorch_unfused_results = benchmark_function(pytorch_unfused_baseline, A, B, num_runs, warmup_runs)
    print_benchmark_results("PyTorch Unfused", pytorch_unfused_results)
    
    # --- Performance Comparison ---
    print("\n--- PERFORMANCE COMPARISON ---")
    
    # Compare fused vs unfused
    fused_speedup = pytorch_fused_results['avg_time'] / fused_results['avg_time']
    print(f"\nFused kernel speedup: {fused_speedup:.2f}x faster than PyTorch")
    
    # Compare Inferno vs PyTorch for unfused
    unfused_speedup = pytorch_unfused_results['avg_time'] / unfused_results['avg_time']
    print(f"Unfused kernel speedup: {unfused_speedup:.2f}x faster than PyTorch")
    
    # Overall comparison
    if fused_speedup > 1.0:
        print(f"✅ Fused kernel shows {fused_speedup:.2f}x performance improvement!")
    else:
        print(f"⚠️  Fused kernel is {1/fused_speedup:.2f}x slower than PyTorch")
    
    print("\n" + "="*60)

