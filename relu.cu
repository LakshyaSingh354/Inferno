#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>


using namespace std;

__global__ void relu(const float* in, float* out, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_vec4) {
        float4 val = in[idx];

        val.x = val.x > 0 ? val.x : 0;
        val.y = val.y > 0 ? val.y : 0;
        val.z = val.z > 0 ? val.z : 0;
        val.w = val.w > 0 ? val.w : 0;

        out[idx] = val;
    }
}

void relu_launcher(torch::Tensor input, torch::Tensor output){
    int N = input.numel();
    TORCH_CHECK(N % 4 == 0, "Input size must be divisible by 4");
    int N_vec4 = N / 4;

    const float4* in_ptr = reinterpret_cast<float4*>(input.data_ptr<float>());
    float4* out_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());

    int threads = 512;
    int blocks = (N_vec4 + threads - 1) / threads;
    relu_vec4<<<blocks, threads>>>(in_ptr, out_ptr, N_vec4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("relu", &relu_launcher, "Custom ReLU kernel (CUDA)");
}
