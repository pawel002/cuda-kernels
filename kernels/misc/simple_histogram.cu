#include <cuda_runtime.h>

__global__
void hist_kernel(
    const int* input,
    int* histogram,
    int N,
    int num_bins)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    atomicAdd(&histogram[input[i]], 1);
}

void solve(const int* input, int* histogram, int N, int num_bins)
{
    int threads = 256;
    int blocks = (N + threads  - 1) / threads;
    hist_kernel<<<blocks, threads>>>(
        input, histogram, N, num_bins
    );
    cudaDeviceSynchronize();
}
