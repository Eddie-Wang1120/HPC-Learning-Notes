#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define M 32
#define K 16
#define N 32


double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void simpleGEMM(int A[M][K], int B[K][N], int C[M][N]){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            for(int p=0;p<K;p++){
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

__global__ void NaiveGEMM(float* A, float* B, float* C){

    __shared__ float As[M*K];
    __shared__ float Bs[K*N];
    
    #pragma unroll
    for(int i=0;i<M*K;i++) As[i] = A[i];
    #pragma unroll
    for(int i=0;i<K*N;i++) Bs[i] = B[i];
    
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idy = blockDim.y*blockIdx.y+threadIdx.y;

    if(idx<M && idy<N){
        float tmp = 0.0f;
        #pragma unroll
        for(int i=0;i<K;i++){
            tmp += As[idy*K + i]*Bs[idx + i*N];
        }
        C[idx + N*idy] = tmp;
    }
}

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
        A[i] = 1;
    }

    for(int i=0;i<K*N;i++){
        B[i] = i;
    }

    dim3 grid(1);
    dim3 block(M, N);

    double start = seconds();

    cudaMemcpy(gpu_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    NaiveGEMM<<<grid, block>>>(gpu_A, gpu_B, gpu_ref);
    cudaMemcpy(cpu_ref, gpu_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    double Elpas = seconds() - start;
    printf("using %lf\n", Elpas);

    for(int i=0;i<10;i++){
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
