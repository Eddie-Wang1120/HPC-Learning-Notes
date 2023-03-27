#include <stdio.h>

#define SMEMDIM 16

__device__ int warpReduce(int mySum){
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2);
    mySum += __shfl_xor(mySum, 1);
    return mySum;
}

__global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int smem[SMEMDIM];
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=n) return;

    int mySum = g_idata[idx];

    int laneIdx = threadIdx.x%warpSize;
    int warpIdx = threadIdx.x/warpSize;

    mySum = warpReduce(mySum);

    if(laneIdx==0) smem[warpIdx] = mySum;

    __syncthreads();

    mySum = (threadIdx.x<SMEMDIM)?smem[laneIdx]:0;
    if(warpIdx==0) mySum = warpReduce(mySum);

    if(threadIdx.x==0) g_odata[blockIdx.x] = mySum;
}

void initialData(float* data, int size){
    for(int i=0;i<size;i++){
        data[i] = i;
    }
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    printf("%s starting\n",deviceProp.name);


    int nElem = BDIMX;
    size_t nBytes = nElem*sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);

    float *d_A, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    dim3 block(BDIMX, 1);
    dim3 grid(1, 1);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    test_shfl_broadcast<<<grid, block>>>(d_C, d_A, 4);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i<BDIMX;i++){
        printf("%f ",gpuRef[i]);
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(gpuRef);

    return EXIT_SUCCESS;

}