#include <cuda_runtime.h>

#define TS 16 // tile size

__global__ void matrix_multiplication_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ A_rows[TS][];
    __shared__ B_cols[TS][];

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = sum;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid(
        (K + TS - 1) / TS, 
        (M + TS - 1) / TS
    );
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
