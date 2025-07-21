#include <cuda_runtime.h>

using namespace std;

__global__ void relu_activation(float* __restrict__ data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = val > 0.0f ? val : 0.0f;
    }
}

extern "C" void launch_relu_activation(float* data, int size, int threads, int blocks) {
    relu_activation<<<blocks, threads>>>(data, size);
}

