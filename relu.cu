#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>


using namespace std;

__global__ void relu(const float* in, float* out, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N){
        out[idx] = in[idx] > 0 ? in[idx] : 0;
    }
}

void relu_launcher(torch::Tensor input, torch::Tensor output){
    int N = input.numel();
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu<<<blocks, threads>>>(in_ptr, out_ptr, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("relu", &relu_launcher, "Custom ReLU kernel (CUDA)");
}

int main(){
    const int N = 16;
    float in_h[N] = {-1.0, 2.5, -0.3, 4.2, 0.0, -2.2, 1.1, 7.7,
                    -5.0, 0.3, 0.8, -9.9, 2.2, 3.3, -4.4, 5.5};
    float out_h[N];
    float *in_d, *out_d;

    cudaMalloc(&in_d, N * sizeof(float));
    cudaMalloc(&out_d, N * sizeof(float));

    cudaMemcpy(in_d, in_h, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 4;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu<<<blocks, threadsPerBlock>>>(in_d, out_d, N);

    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "ReLU Output: ";
    for (int i = 0; i < N; ++i){
        cout << out_h[i] << " ";
    }
    cout << endl;

    cudaFree(in_d);
    cudaFree(out_d);

    return 0;

}