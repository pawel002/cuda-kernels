#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

constexpr int THREADS = 1024;

template <int TS>
__global__
void dot_kernel(
    const half* A,
    const half* B,
    float* partial_d,
    int N)
{
    __shared__ float shared[TS];
    int i = 2 * TS * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (i < N)
        sum += __half2float(A[i]) * __half2float(B[i]);

    if (i + TS < N)
        sum += __half2float(A[i + TS]) * __half2float(B[i + TS]);

    shared[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (unsigned stride = TS / 2; stride >= 32; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    sum = shared[tid];
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (tid == 0) partial_d[blockIdx.x] = sum;
}

template<int TS>
__global__
void reduce_add_kernel(
    const float* input,
    half* result,
    int N)
{
    __shared__ float shared[TS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    for (int k = 0; i + k * TS < N; k++)
        sum += input[i + k * TS];

    shared[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (unsigned stride = TS / 2; stride >= 32; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    sum = shared[tid];
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (tid == 0) *result = __float2half(sum);
}

void solve(const half* A, const half* B, half* result, int N) 
{
    int blocks = (N + 2 * THREADS - 1) / (2 * THREADS);
    float *partial_d;
    cudaMalloc(&partial_d, blocks * sizeof(float));

    dot_kernel<THREADS><<<blocks, THREADS>>>(
        A, B, partial_d, N
    );
    cudaDeviceSynchronize();

    reduce_add_kernel<THREADS><<<1, THREADS>>>(
        partial_d, result, blocks
    );
    cudaDeviceSynchronize();
}
