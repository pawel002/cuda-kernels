#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

__device__ float d_max;
__device__ float d_norm;

__device__ static float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float mdata[];
    mdata[tid] = (idx < N) ? input[idx] : -FLT_MAX;

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && tid + s < blockDim.x)
            mdata[tid] = fmaxf(mdata[tid], mdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMaxFloat(&d_max, mdata[0]);
}

__global__ void reduction_kernel(const float* input, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    sdata[tid] = (idx < N) ? __expf(input[idx] - d_max) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && tid + s < blockDim.x)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(&d_norm, sdata[0]);
}

__global__ void softmax_kernel(const float* input, float* output, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        output[idx] = __expf(input[idx] - d_max) / d_norm;
}

void solve(const float* input, float* output, int N)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float init_max = -FLT_MAX;
    float init_norm = 0.0f;
    cudaMemcpyToSymbol(d_max, &init_max, sizeof(float));
    cudaMemcpyToSymbol(d_norm, &init_norm, sizeof(float));

    max_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, N);
    cudaDeviceSynchronize();

    reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, N);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}


