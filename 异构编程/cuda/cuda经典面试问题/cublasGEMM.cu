#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>

const int M = 32;//A_ROW 
const int K = 16;//A_COL B_ROW
const int N = 32;//B_COL

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, dev);
    cudaSetDevice(dev);

    float* h_A = (float *)malloc(M*K*sizeof(float));
    float* h_B = (float *)malloc(K*N*sizeof(float));
    float* cpuref = (float *)malloc(M*N*sizeof(float));
    
    for(int i=0;i<M*K;i++){
        h_A[i] = i%10;
    }

    for(int i=0;i<M*K;i++){
        h_B[i] = i;
    }

    // for(int i=0;i<M*K;i++){
    //     printf("%f ", h_A[i]);
    //     if((i+1)%N==0) printf("\n");
    // }

    // printf("\n");

    // for(int i=0;i<K*N;i++){
    //     printf("%f ", h_B[i]);
    //     if((i+1)%N==0) printf("\n");
    // }

    printf("\n");
    float* d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, M*K*sizeof(float));
    cudaMalloc((float **)&d_B, K*N*sizeof(float));
    cudaMalloc((float **)&d_C, M*N*sizeof(float));

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    double start = seconds();
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    float a = 1, b = 0;
    
    cublasSgemm(
        blas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &a,
        d_B,
        N,
        d_A,
        K,
        &b,
        d_C,
        N
    );

    cudaMemcpy(cpuref, d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    double Elpas = seconds() - start;
    printf("using %lf\n", Elpas);

    for(int i=0;i<16;++i){
        printf("%f ", cpuref[i]);
        if((i+1)%N==0) printf("\n");
    };
    printf("\n");

    free(cpuref);
    free(h_A);
    free(h_B);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return 0;


}