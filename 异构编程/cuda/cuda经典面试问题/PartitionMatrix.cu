#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <stdlib.h>

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

template<int BLOCK_SIZE>
__global__ void Partition_Matrix(float* A, float* B, float* C, const int M, const int K, const int N){

    float accu = 0;

    for(int tileIdx = 0; tileIdx < K / blockDim.x; tileIdx ++) {
        __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];
        
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = tileIdx * blockDim.x + threadIdx.x;
        As[threadIdx.y*BLOCK_SIZE+threadIdx.x] = A[i*K+j];
        Bs[threadIdx.x*BLOCK_SIZE+threadIdx.y] = B[j*N+i];
        __syncthreads();
        #pragma unroll
        for(int k = 0 ; k < blockDim.x ; k ++ ) {
            accu = accu + As[threadIdx.y*BLOCK_SIZE+k] * Bs[k*BLOCK_SIZE+threadIdx.x];
        }
        __syncthreads();
    }

    int i = blockIdx.x * blockDim.x + threadIdx.y; 
    int j = blockIdx.y * blockDim.y + threadIdx.x;
    C[i*N+j] = accu;       
}

const int M = 32;
const int K = 16;
const int N = 32;
const int BLOCK_SIZE = 4;

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, dev);
    cudaSetDevice(dev);
    printf("%s starting now ...\n", devprop.name);
    printf("SM数量： %d\n", devprop.multiProcessorCount);
    printf("每个线程块共享内存大小：%fKB\n", devprop.sharedMemPerBlock/1024.0);
    printf("每个线程块最大线程数：%d\n",devprop.maxThreadsPerBlock);
    printf("每个SM最大线程数：%d\n",devprop.maxThreadsPerMultiProcessor);
    printf("每个SM最大线程束数：%d\n",devprop.maxThreadsPerMultiProcessor/32);

    float* A = (float*)malloc(M*K*sizeof(float));
    float* B = (float*)malloc(K*N*sizeof(float));
    float* cpu_ref = (float*)malloc(M*N*sizeof(float));

    float* gpu_A, *gpu_B, *gpu_ref;
    cudaMalloc((float **)&gpu_A, M*K*sizeof(float));
    cudaMalloc((float **)&gpu_B, K*N*sizeof(float));
    cudaMalloc((float **)&gpu_ref, M*N*sizeof(float));

    for(int i=0;i<M*K;i++){
        A[i] = i%10;
    }

    for(int i=0;i<K*N;i++){
        B[i] = i;
    }

    dim3 grid(M/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    double start = seconds();

    cudaMemcpy(gpu_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    Partition_Matrix<BLOCK_SIZE><<<grid, block>>>(gpu_A, gpu_B, gpu_ref, M, K, N);
    cudaMemcpy(cpu_ref, gpu_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    double Elpas = seconds() - start;
    printf("using %lf\n", Elpas);

    for(int i=0;i<16;i++){
        printf("%f ", cpu_ref[i]);
    }
    printf("\n");

    free(cpu_ref);
    free(A);
    free(B);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_ref);

    cudaDeviceReset();

    return 0;

}