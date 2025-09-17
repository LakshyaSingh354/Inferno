import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path so we can import the compiler
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from inferno import compile_model

# ================================================================================
# Benchmark Configuration
# ================================================================================
# A list of matrix shapes to test. (M, K, N) for C(M,N) = A(M,K) @ B(K,N)
MATRIX_SIZES = [
    (32, 32, 32),
    (64, 32, 64),
    (64, 64, 64),
    (64, 128, 64),
    (128, 256, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
]

# Number of warm-up and measurement runs
WARMUP_RUNS = 10
MEASURE_RUNS = 100

# ================================================================================
# Model Definitions
# ================================================================================

@compile_model(example_inputs=[torch.randn(256, 256, device='cuda')], kernel_filepath='src/fused_kernel.cu')
class FusionModel(nn.Module):
    """
    Model with the pattern that can be fused: matmul followed by relu
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        
    def forward(self, x):
        # This pattern: matmul + relu can be fused
        x = torch.matmul(x, self.weight)
        x = F.relu(x)
        return x

class NonFusionModel(nn.Module):
    """
    Model without the fusion pattern: just matmul, no relu
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        
    def forward(self, x):
        # This pattern: just matmul, no relu - cannot be fused
        x = torch.matmul(x, self.weight)
        return x

def benchmark_size(m, k, n):
    """
    Runs a benchmark for a single matrix size and returns the average latencies.
    """
    print(f"\n--- Benchmarking size (M, K, N) = ({m}, {k}, {n}) ---")

    # Create random input tensor on the GPU
    try:
        input_tensor = torch.randn(m, k, device='cuda', dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("    Skipping due to CUDA Out of Memory.")
        return None, None, None

    # Create models
    fusion_model = FusionModel(k, k, n).cuda()
    non_fusion_model = NonFusionModel(k, k, n).cuda()   

    # --- 1. Non-Fusion Model Benchmark (PyTorch Vanilla) ---
    
    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        _ = non_fusion_model(input_tensor)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(MEASURE_RUNS):
        output_non_fusion = non_fusion_model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    non_fusion_time_ms = start_event.elapsed_time(end_event) / MEASURE_RUNS
    print(f"    Non-Fusion Model (PyTorch) Average Latency: {non_fusion_time_ms:.6f} ms")

    # --- 2. Compiled Fusion Model Benchmark ---

    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        _ = fusion_model(input_tensor)
    torch.cuda.synchronize()

    # Timed runs
    start_event.record()
    for _ in range(MEASURE_RUNS):
        output_fusion = fusion_model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()

    fusion_time_ms = start_event.elapsed_time(end_event) / MEASURE_RUNS
    print(f"    Compiled Fusion Model Average Latency: {fusion_time_ms:.6f} ms")
    speedup = non_fusion_time_ms / fusion_time_ms
    print(f"    Speedup: {speedup:.2f}x")

    # --- 3. Verification ---
    try:
        # Compare fusion model output with non-fusion model + manual relu
        expected_output = F.relu(torch.matmul(input_tensor, fusion_model.weight))
        is_close = torch.allclose(output_fusion, expected_output, atol=1e-3)
        if not is_close:
            print("    !! WARNING: Results are NOT close. Check implementation.")
        else:
            print("    Verification: PASSED")
    except Exception as e:
        print(f"    Verification failed with error: {e}")
        is_close = False

    return non_fusion_time_ms, fusion_time_ms, is_close


def main():
    """
    Main function to run the full benchmark suite and plot results.
    """
    results = []
    for m, k, n in MATRIX_SIZES:
        non_fusion_ms, fusion_ms, verified = benchmark_size(m, k, n)
        if non_fusion_ms is not None:
            results.append({
                "Size": f"{m}x{k}x{n}",
                "Non-Fusion (ms)": non_fusion_ms,
                "Fusion (ms)": fusion_ms,
                "Speedup": non_fusion_ms / fusion_ms if fusion_ms > 0 else float('inf'),
                "Verified": "Yes" if verified else "No"
            })

    # --- Display Results Table ---
    if not results:
        print("\nNo benchmarks were successfully run. Exiting.")
        return
        
    df = pd.DataFrame(results)
    print("\n\n" + "="*80)
    print(" " * 25 + "BENCHMARKING RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # --- Plot Results ---
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart for latencies
    bar_width = 0.35
    index = range(len(df))
    
    bar1 = ax1.bar(index, df["Non-Fusion (ms)"], bar_width, label='Non-Fusion (PyTorch)', color='cornflowerblue')
    bar2 = ax1.bar([i + bar_width for i in index], df["Fusion (ms)"], bar_width, label='Compiled Fusion', color='orangered')

    ax1.set_xlabel('Matrix Size (M x K x N)', fontweight='bold')
    ax1.set_ylabel('Average Latency (ms)', fontweight='bold')
    ax1.set_title('Compiled Fusion vs. Non-Fusion: Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(df["Size"], rotation=45, ha="right")
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Line chart for speedup on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot([i + bar_width / 2 for i in index], df["Speedup"], color='green', linestyle='--', marker='o', linewidth=2, label='Speedup (X)')
    ax2.set_ylabel('Speedup (Non-Fusion / Fusion)', fontweight='bold', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig("results/inferno_benchmark.png", dpi=300)
    print("\nBenchmark plot saved to 'results/inferno_benchmark.png'")
    plt.show()


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a machine with CUDA support.")
        exit(1)
        
    main()

