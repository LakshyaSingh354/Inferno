import torch
import pandas as pd
import matplotlib.pyplot as plt
import inferno_fused 
# ================================================================================
# Benchmark Configuration
# ================================================================================
# A list of matrix shapes to test. (M, K, N) for C(M,N) = A(M,K) @ B(K,N)
MATRIX_SIZES = [
    (64, 128, 256),
    (128, 256, 512),
    (256, 512, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

# Number of warm-up and measurement runs
WARMUP_RUNS = 10
MEASURE_RUNS = 100

def benchmark_size(m, k, n):
    """
    Runs a benchmark for a single matrix size and returns the average latencies.
    """
    print(f"\n--- Benchmarking size (M, K, N) = ({m}, {k}, {n}) ---")

    # Create random input tensors on the GPU
    try:
        A = torch.randn(m, k, device='cuda', dtype=torch.float32)
        B = torch.randn(k, n, device='cuda', dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        print("    Skipping due to CUDA Out of Memory.")
        return None, None, None

    # --- 1. PyTorch Vanilla Benchmark ---
    
    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        _ = torch.relu(torch.matmul(A, B))
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(MEASURE_RUNS):
        C_pytorch = torch.relu(torch.matmul(A, B))
    end_event.record()
    torch.cuda.synchronize()
    
    pytorch_time_ms = start_event.elapsed_time(end_event) / MEASURE_RUNS
    print(f"    PyTorch Average Latency: {pytorch_time_ms:.6f} ms")


    # --- 2. Inferno Fused Kernel Benchmark ---

    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        _ = inferno_fused.fused_gemm_relu(A, B)
    torch.cuda.synchronize()

    # Timed runs
    start_event.record()
    for _ in range(MEASURE_RUNS):
        C_inferno = inferno_fused.fused_gemm_relu(A, B)
    end_event.record()
    torch.cuda.synchronize()

    inferno_time_ms = start_event.elapsed_time(end_event) / MEASURE_RUNS
    print(f"    Inferno Average Latency: {inferno_time_ms:.6f} ms")

    # --- 3. Verification ---
    try:
        is_close = torch.allclose(C_pytorch, C_inferno, atol=1e-5)
        if not is_close:
            print("    !! WARNING: Results are NOT close. Check implementation.")
        else:
            print("    Verification: PASSED")
    except Exception as e:
        print(f"    Verification failed with error: {e}")

    return pytorch_time_ms, inferno_time_ms, is_close


def main():
    """
    Main function to run the full benchmark suite and plot results.
    """
    results = []
    for m, k, n in MATRIX_SIZES:
        pytorch_ms, inferno_ms, verified = benchmark_size(m, k, n)
        if pytorch_ms is not None:
            results.append({
                "Size": f"{m}x{k}x{n}",
                "PyTorch (ms)": pytorch_ms,
                "Inferno (ms)": inferno_ms,
                "Speedup": pytorch_ms / inferno_ms if inferno_ms > 0 else float('inf'),
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
    
    bar1 = ax1.bar(index, df["PyTorch (ms)"], bar_width, label='PyTorch Vanilla', color='cornflowerblue')
    bar2 = ax1.bar([i + bar_width for i in index], df["Inferno (ms)"], bar_width, label='Inferno Fused Kernel', color='orangered')

    ax1.set_xlabel('Matrix Size (M x K x N)', fontweight='bold')
    ax1.set_ylabel('Average Latency (ms)', fontweight='bold')
    ax1.set_title('Inferno Fused Kernel vs. PyTorch Vanilla: Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(df["Size"], rotation=45, ha="right")
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Line chart for speedup on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot([i + bar_width / 2 for i in index], df["Speedup"], color='green', linestyle='--', marker='o', linewidth=2, label='Speedup (X)')
    ax2.set_ylabel('Speedup (PyTorch / Inferno)', fontweight='bold', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig("results/inferno_benchmark.png", dpi=300)
    print("\nBenchmark plot saved to 'results/inferno_benchmark.png'")
    plt.show()


if __name__ == "__main__":
    # Ensure the compiled extension is available
    try:
        # A small check to see if the op exists
        _ = inferno_fused.fused_gemm_relu
    except AttributeError:
        print("="*80)
        print("ERROR: Could not find 'fused_gemm_relu' in the compiled 'inferno_fused' module.")
        print("Please ensure you have compiled the C++/CUDA code successfully using 'python setup.py install'.")
        print("="*80)
        exit()
        
    main()

