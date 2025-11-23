#include <cuda_runtime.h>
#include <cstdio>

#define TILE_DIM 32

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    // loop over tiles of K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        int A_col = t * TILE_DIM + threadIdx.x;
        int B_row = t * TILE_DIM + threadIdx.y;

        // load A tile
        if (row < M && A_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile
        if (B_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // compute partial dot product for this tile
        for (int k = 0; k < TILE_DIM; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);

    matmul_tiled<<<grid, block>>>(A, B, C, M, K, N);
}
