#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

struct SubarrayIndexMap3D
{
    int row_stride;
    int depth_stride;
    int S_DEP, S_ROW, S_COL;

    int sub_cols;
    int sub_slice_area;

    SubarrayIndexMap3D(int M, int K, 
                       int s_dep, int s_row, int s_col, 
                       int e_row, int e_col) 
        : S_DEP(s_dep), S_ROW(s_row), S_COL(s_col) 
    {
        row_stride = K;
        depth_stride = M * K;

        sub_cols = e_col - s_col + 1;
        int sub_rows = e_row - s_row + 1;
        
        sub_slice_area = sub_rows * sub_cols;
    }

    __host__ __device__
    int operator()(int idx) const {
        
        int local_d = idx / sub_slice_area;
        int rem     = idx % sub_slice_area;
        
        int local_r = rem / sub_cols;
        int local_c = rem % sub_cols;

        int global_d = S_DEP + local_d;
        int global_r = S_ROW + local_r;
        int global_c = S_COL + local_c;

        return (global_d * depth_stride) + (global_r * row_stride) + global_c;
    }
};

extern "C" void solve(const int* input, int* output, 
                      int N, int M, int K, 
                      int S_DEP, int E_DEP, 
                      int S_ROW, int E_ROW, 
                      int S_COL, int E_COL) 
{
    int num_deps = E_DEP - S_DEP + 1;
    int num_rows = E_ROW - S_ROW + 1;
    int num_cols = E_COL - S_COL + 1;
    
    int total_elements = num_deps * num_rows * num_cols;
    thrust::device_ptr<const int> d_input(input);
    thrust::counting_iterator<int> counter(0);

    SubarrayIndexMap3D index_mapper(M, K, S_DEP, S_ROW, S_COL, E_ROW, E_COL);

    auto transform_iter = thrust::make_transform_iterator(counter, index_mapper);
    auto subarray_iter = thrust::make_permutation_iterator(d_input, transform_iter);

    int sum = thrust::reduce(subarray_iter, subarray_iter + total_elements);
    cudaMemcpy(output, &sum, sizeof(int), cudaMemcpyHostToDevice);
}