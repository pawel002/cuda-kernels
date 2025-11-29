#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

void solve(const int* input, int* output, int N, int S, int E) {

    int* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // hacky way for now :P
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input+S, output, E-S+1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input+S, output, E-S+1);
}