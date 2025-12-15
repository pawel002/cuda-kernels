#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row >= M || col >= N)return;
    float sum = 0.0f;
    for(int k = 0; k < K; ++k){
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]);
        sum += a * b;
    }
    float c_val = (beta != 0.0f) ? __half2float(C[row * N + col])*beta :0.0f;
    C[row * N + col] = __float2half_rn(c_val + alpha * sum);

}

void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int TILE_SIZE = 16;
    dim3 block_size(TILE_SIZE, 32);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (M + 32 - 1) / 32);
    gemm_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
