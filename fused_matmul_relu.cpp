#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>

extern "C" void launch_relu_activation(float* data, int size, int threads, int blocks);

void matmul_relu(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "All tensors must be on CUDA.");

    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        B.data_ptr<float>(), n,
        A.data_ptr<float>(), k,
        &beta,
        C.data_ptr<float>(), n
    );

    int size = m * n;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    launch_relu_activation(C.data_ptr<float>(), size, threads, blocks);

    cublasDestroy(handle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_relu", &matmul_relu, "Fused MatMul + ReLU (CUDA)");
}