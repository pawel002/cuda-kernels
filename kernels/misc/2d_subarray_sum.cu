#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

struct SubarrayIndexMap
{
    int M; 
    int S_ROW; 
    int S_COL; 
    int sub_width;

    SubarrayIndexMap(int m, int s_row, int s_col, int e_col) 
        : M(m), S_ROW(s_row), S_COL(s_col) {
        sub_width = e_col - s_col + 1;
    }

    __host__ __device__
    int operator()(int idx) const {
        int local_r = idx / sub_width;
        int local_c = idx % sub_width;

        int global_r = S_ROW + local_r;
        int global_c = S_COL + local_c;

        return (global_r * M) + global_c;
    }
};

void solve(const int* input, int* output, 
    int N, int M, 
    int S_ROW, int E_ROW, 
    int S_COL, int E_COL) 
{
    int num_rows = E_ROW - S_ROW + 1;
    int num_cols = E_COL - S_COL + 1;
    int total_elements = num_rows * num_cols;

    thrust::device_ptr<const int> d_input(input);
    thrust::counting_iterator<int> counter(0);

    SubarrayIndexMap index_mapper(M, S_ROW, S_COL, E_COL);
    auto transform_iter = thrust::make_transform_iterator(counter, index_mapper);
    auto subarray_iter  = thrust::make_permutation_iterator(d_input, transform_iter);

    int sum = thrust::reduce(subarray_iter, subarray_iter + total_elements);
    cudaMemcpy(output, &sum, sizeof(int), cudaMemcpyHostToDevice);
}