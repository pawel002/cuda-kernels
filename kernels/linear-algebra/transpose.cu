#include <cuda_runtime.h>

#define TS 16 // tile size

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = TS * blockIdx.x + threadIdx.x;
    int row = TS * blockIdx.y + threadIdx.y;

    for (int i = row; i < row + TS; i++)
    {
        if (i >= rows) break;
        for (int j = col; j < col + TS; j++)
        {
            if (j >= cols) break;
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid((cols + TS - 1) / TS,
                       (rows + TS - 1) / TS);

    matrix_transpose_kernel<<<blocksPerGrid, TS>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}