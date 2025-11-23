#include <cuda_runtime.h>

__global__ void integrate(const float* y, float* out, float dx, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) atomicAdd(out, dx * y[tid]);
}

void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    const int threads = 256;
    const int blocks  = (n_samples + threads - 1) / threads;
    const float dx = (b - a) / (float)n_samples;

    cudaMemset(result, 0, sizeof(float));

    integrate<<<blocks, threads>>>(y_samples, result, dx, n_samples);
    cudaDeviceSynchronize();
}