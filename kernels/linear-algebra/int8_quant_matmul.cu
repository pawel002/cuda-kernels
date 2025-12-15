#include <cuda_runtime.h>

#define tile 32

__device__ __forceinline__ int8_t clamp(int acc,float scale_A, float scale_B,float scale_C,int zero_point_C){
    float val=acc*scale_A*scale_B/scale_C;
    val+=zero_point_C;
    int val2=lroundf(val);
    if(val2>127) return 127;
    if(val2<-128) return -128;
    return val2;
}

__global__ void matmul(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,float scale_A, float scale_B, float scale_C,int zero_point_A, int zero_point_B,int zero_point_C){
    int by=blockIdx.y;
    int bx=blockIdx.x;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int ga_m=by*tile+ty;
    int gb_n=bx*tile+tx;
    __shared__ int8_t sa[tile][tile];
    __shared__ int8_t sb[tile][tile];
    int acc=0;
    for(int k=0;k<(K+tile-1)/tile;k++){
        if(ga_m<M&&k*tile+tx<K){
            sa[ty][tx]=A[ga_m*K+k*tile+tx];
        }
        else{
            sa[ty][tx]=0;
        }
        if(gb_n<N&&k*tile+ty<K){
            sb[ty][tx]=B[(k*tile+ty)*N+gb_n];
        }
        else{
            sb[ty][tx]=0;
        }
        __syncthreads();
        for(int tk=0;tk<tile;tk++){
            acc+=(sa[ty][tk]-zero_point_A)*(sb[tk][tx]-zero_point_B);
        }
        __syncthreads();
    }
    if(ga_m<M&&gb_n<N){
        C[ga_m*N+gb_n]=clamp(acc,scale_A,scale_B,scale_C,zero_point_C);
    }

}

void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    dim3 block(tile,tile);
    dim3 grid((N+tile-1)/tile,(M+tile-1)/tile);
    matmul<<<grid,block>>>(A,B,C,M,N,K,scale_A,scale_B,scale_C,zero_point_A,zero_point_B,zero_point_C);
} 