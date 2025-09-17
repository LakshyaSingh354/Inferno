#include <torch/extension.h>
#include <cuda_runtime.h>

// ================================================================================
// Optimized Fused GEMM + ReLU Kernel (using shared memory and bank conflict avoidance)
// ================================================================================
__global__ void gemm_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    extern __shared__ float shared_mem[];
    const int block_size = blockDim.x;
    
    const int row = blockIdx.y * block_size + threadIdx.y;
    const int col = blockIdx.x * block_size + threadIdx.x;
    
    float* As = shared_mem;
    float* Bs = shared_mem + block_size * block_size;
    
    float sum = 0.0f;

    // Loop over tiles in K dimension
    for (int tile_idx = 0; tile_idx < K; tile_idx += block_size) {
        // Load A tile: A[row, tile_idx + threadIdx.x]
        int A_col = tile_idx + threadIdx.x;
        if (row < M && A_col < K) {
            As[threadIdx.y * block_size + threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        

        // Load B tile: B[tile_idx + threadIdx.y, col]
        int B_row = tile_idx + threadIdx.y;
        if (B_row < K && col < N) {
            Bs[threadIdx.y * block_size + threadIdx.x] = B[B_row * N + col];
        } else {
            Bs[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < block_size && (tile_idx + k) < K; ++k) {
            sum += As[threadIdx.y * block_size + k] 
                 * Bs[k * block_size + threadIdx.x];
        }

        __syncthreads();
    }

    // Apply ReLU and store result
    if (row < M && col < N) {
        C[row * N + col] = fmaxf(sum, 0.0f);
    }
}

// ================================================================================
// C++ Host Function
// ================================================================================
void fused_gemm_relu_forward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Configure kernel parameters (optimal for most modern GPUs)
    const int block_size = 16;
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, 
              (M + block_size - 1) / block_size);

    // Calculate shared memory size
    size_t shared_mem_size = (2 * block_size * block_size) * sizeof(float);

    // Launch optimized kernel
    gemm_relu_kernel<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
}