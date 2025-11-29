#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

void solve(const int* input, int* output, int N, int S, int E) {
    thrust::device_ptr<const int> dev_ptr(input);
    int sum = thrust::reduce(dev_ptr + S, dev_ptr + E + 1);
    cudaMemcpy(output, &sum, sizeof(int), cudaMemcpyHostToDevice);
}