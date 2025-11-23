#include <stdio.h>
#include <cuda_runtime.h>

__device__
void warp_reduce(volatile float* input, int tid)
{
    input[tid] += input[tid + 32];
    input[tid] += input[tid + 16];
    input[tid] += input[tid + 8];
    input[tid] += input[tid + 4];
    input[tid] += input[tid + 2];
    input[tid] += input[tid + 1];
}

__global__
void reduce_add(const float* input, float* output, int N)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i   = blockDim.x * (blockIdx.x * 2) + threadIdx.x;

    // fill shared memory
    shared[tid] = (i < N)
        ? input[i] + ((i + blockDim.x < N) ? input[i + blockDim.x] : 0.0f)
        : 0.0f;
    __syncthreads();

    // reduce shared memory
    for (int s = blockDim.x/2; s>32; s >>= 1)
    {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    // last explicit reduction
    if (tid < 32) warp_reduce(shared, tid);
    if (tid == 0) output[blockIdx.x] = shared[0];
}

void solve(const float* input, float* output, int N) 
{  
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    // allocate partials
    float* partials_d;
    cudaMalloc(&partials_d, blocks * sizeof(float));

    // run first reduction explicitely
    reduce_add<<<blocks, threads, shared_memory_size>>>(input, partials_d, N);
    cudaDeviceSynchronize();

    float* partials_h = (float*) malloc(blocks * sizeof(float));
    cudaMemcpy(partials_h, partials_d, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // input size and partials for reduction loop
    int n = blocks;
    blocks = (n + threads - 1) / threads;
    float* partials_d_next;
    cudaMalloc(&partials_d_next, blocks * sizeof(float));

    // reduction loop
    float* temp;
    while (n > 1)
    { 
        reduce_add<<<blocks, threads, shared_memory_size>>>(partials_d, partials_d_next, n);
        cudaDeviceSynchronize();

        // swap pointers
        n = blocks;
        blocks = (n + threads - 1) / threads;
        temp = partials_d;
        partials_d = partials_d_next;
        partials_d_next = temp;
    }

    cudaMemcpy(output, partials_d, sizeof(float), cudaMemcpyDeviceToDevice);
}