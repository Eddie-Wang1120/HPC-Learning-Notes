#include <stdio.h>
#define NSTREAM 8

__global__ void sumArrays(float* A, float* B, float* C, int N){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int n_repeats = 1000;

    if(idx<N){
        for(int i=0;i<n_repeats;i++){
            C[idx] = A[idx]+B[idx];
        }
    }
}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    printf("device %s starting\n", deviceProp.name);

    cudaStream_t *streams = (cudaStream_t *)malloc(NSTREAM*sizeof(cudaStream_t));
    for(int i=0;i<NSTREAM;i++){
        cudaStreamCreate(&(streams[i]));
    }

    int nElem = 1024;
    int nBytes = nElem*sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);

    cudaHostAlloc((float **)&gpuRef, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((float **)&hostRef, nBytes, cudaHostAllocDefault);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    int iElem = nElem / NSTREAM;
    int iBytes = iElem*sizeof(float);

    dim3 grid(1);
    dim3 block(1);

    for(int i = 0;i<NSTREAM;i++){
        int ioffset = i*iElem;
        cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes, cudaMemcpyHostToDevice, streams[i]);
        sumArrays<<<grid, block, 0, streams[i]>>>(&d_A[ioffset], &d_B[ioffset], &d_C[ioffset], iElem);
    }


    for(int i = 0;i<NSTREAM;i++){
        int ioffset = i*iElem;
        cudaMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    free(h_A);
    free(h_B);    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    cudaDeviceReset();

    return EXIT_SUCCESS;

}