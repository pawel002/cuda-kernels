#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= halfN) return;

    float x1 = input[i];
    float x2 = input[i + halfN];

    output[i] = x2 * x1 / (1 + __expf(-x1));
}

void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}