/*
================================================================================
 Fused Kernel: cuBLAS GEMM + ReLU for the Inferno Compiler
================================================================================

This file contains the complete implementation for a logically fused kernel that
performs a high-performance matrix multiplication using NVIDIA's cuBLAS library,
followed immediately by a custom ReLU activation kernel.

The "fusion" here is logical: we execute two separate GPU operations back-to-back
on the same CUDA stream from a single C++ function call. This minimizes CPU-GPU
synchronization and avoids writing the intermediate result of the MatMul back to
system RAM, which is a major performance win.

This file is broken down into three main sections:
1.  The custom CUDA kernel for the ReLU activation.
2.  The C++ host function that orchestrates the cuBLAS call and the ReLU kernel launch.
3.  The pybind11 wrapper to expose this C++ function to Python, allowing it to
    be called directly with PyTorch tensors.

================================================================================
*/

#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// Error checking macros
#define CHECK_CUDA(x) AT_ASSERTM(x.ok(), #x " failed: " #x)
#define CHECK_CUBLAS(x) AT_ASSERTM((x) == CUBLAS_STATUS_SUCCESS, #x " failed!")

// ================================================================================
// SECTION 1: The Custom CUDA ReLU Kernel
// ================================================================================
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// ================================================================================
// SECTION 2: The C++ Host Orchestrators
// ================================================================================

// This is the core C++ function that will be called from Python. It orchestrates
// the two GPU operations.
void fused_gemm_relu_forward_cuda(
    cublasHandle_t handle,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {

    // A is (M, K), B is (K, N), C is (M, N)
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float alpha = 1.0f;
    const float beta = 0.0f;


    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B.data_ptr<float>(), N,
                             A.data_ptr<float>(), K,
                             &beta,
                             C.data_ptr<float>(), N));

    //Launch the custom ReLU kernel ---
    const int total_elements = M * N;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks_per_grid, threads_per_block>>>(C.data_ptr<float>(), total_elements);
    
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "ReLU kernel launch failed");
}


// --- DEBUG FUNCTION 1: GEMM Only ---
void debug_gemm_only_cuda(
    cublasHandle_t handle,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B.data_ptr<float>(), N,
                             A.data_ptr<float>(), K,
                             &beta,
                             C.data_ptr<float>(), N));
}

// --- DEBUG FUNCTION 2: ReLU Only ---
void debug_relu_only_cuda(torch::Tensor T) {
    const int total_elements = T.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks_per_grid, threads_per_block>>>(T.data_ptr<float>(), total_elements);
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "ReLU kernel launch failed");
}


// ================================================================================
// SECTION 3: The Pybind11 Wrapper
// ================================================================================

cublasHandle_t get_cublas_handle() {
    static bool initialized = false;
    static cublasHandle_t handle;
    if (!initialized) {
        CHECK_CUBLAS(cublasCreate(&handle));
        initialized = true;
    }
    return handle;
}

// Wrapper for the main fused function
torch::Tensor fused_gemm_relu(torch::Tensor A, torch::Tensor B) {
    AT_ASSERTM(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    AT_ASSERTM(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    AT_ASSERTM(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");

    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    fused_gemm_relu_forward_cuda(get_cublas_handle(), A, B, C);
    return C;
}

// Wrapper for the GEMM-only debug function
torch::Tensor debug_gemm_only(torch::Tensor A, torch::Tensor B) {
    AT_ASSERTM(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    AT_ASSERTM(A.size(1) == B.size(0), "Matrix dimensions mismatch");
    AT_ASSERTM(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");

    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    debug_gemm_only_cuda(get_cublas_handle(), A, B, C);
    return C;
}

// Wrapper for the ReLU-only debug function
torch::Tensor debug_relu_only(torch::Tensor T) {
    AT_ASSERTM(T.is_cuda(), "Input must be a CUDA tensor");
    // We operate in-place for this debug function, so we clone the input
    // to avoid modifying the original tensor passed from Python.
    auto T_out = T.clone();
    debug_relu_only_cuda(T_out);
    return T_out;
}


// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_relu", &fused_gemm_relu, "Fused GEMM + ReLU forward (CUDA)");
    m.def("debug_gemm_only", &debug_gemm_only, "DEBUG: GEMM only (CUDA)");
    m.def("debug_relu_only", &debug_relu_only, "DEBUG: ReLU only (CUDA)");
}
