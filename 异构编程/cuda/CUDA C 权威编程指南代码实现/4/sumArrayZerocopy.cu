#include <stdio.h>

void initialData(float* data, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        data[i] = (float)( rand() & 0xFF)/10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i]-gpuRef[i])>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if(match) printf("match!\n");
    return;

}

void sumArrayOnHost(float* h_a, float* h_b, float* h_c, const size_t size){
    for(int i=0;i<size;i++){
        h_c[i] = h_a[i] + h_b[i];
    }
}

__global__ void sumArrayOnGPU(float* d_a, float* d_b, float* d_c){
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    //check if support mapped memory
    if(!deviceProp.canMapHostMemory){
        printf("Device %d does not support", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    int ipower = 10;
    if(argc>1) ipower = atoi(argv[1]);

    int nElem = 1<<ipower;
    size_t nbytes = nElem * sizeof(float);

    float *h_a = (float *)malloc(nbytes);
    float *h_b = (float *)malloc(nbytes);
    float *hostRef = (float *)malloc(nbytes);
    float *gpuRef = (float *)malloc(nbytes);

    initialData(h_a, nElem);
    initialData(h_b, nElem);
    memset(hostRef, 0, nbytes);
    memset(gpuRef, 0, nbytes);

    sumArrayOnHost(h_a, h_b, hostRef, nElem);

    float *d_a, *d_b, *d_c;
    cudaMalloc((float **)&d_a, nbytes);
    cudaMalloc((float **)&d_b, nbytes);
    cudaMalloc((float **)&d_c, nbytes);

    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);
    
    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1)/block.x);
    
    sumArrayOnGPU<<<grid, block>>>(d_a, d_b, d_c);
    cudaMemcpy(gpuRef, d_c, nbytes, cudaMemcpyDeviceToHost);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    //part2: using zerocopy memory for array A and B
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&h_a, nbytes, flags);
    cudaHostAlloc((void **)&h_b, nbytes, flags);

    initialData(h_a, nElem);
    initialData(h_b, nElem);
    memset(hostRef, 0 ,nbytes);
    memset(gpuRef, 0 , nbytes);

    //pass the pointer to device
    cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0);
    cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0);

    sumArrayOnHost(h_a, h_b, hostRef, nElem);
    sumArrayOnGPU<<<grid, block>>>(d_a, d_b, d_c);
    cudaMemcpy(gpuRef, d_c, nbytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}