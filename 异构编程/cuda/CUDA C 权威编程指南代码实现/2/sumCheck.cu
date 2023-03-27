#include <cstdio>
#include <time.h>
#include <sys/time.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void sumArraysCPU(float *A, float *B, float *C, const int N){
    for(int i=0;i<N;i++){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysGPU(float *A, float *B, float *C){
    int i = threadIdx.x;
    C[i] = B[i] + A[i];
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        ip[i] = (float)( rand() & 0xFF)/10.0f;
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

int main(int argc, char **argv){
    //set up device
    int dev = 0;
    cudaSetDevice(dev);

    //set up data size of vectors
    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);

    //malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    double iStart,iElaps;

    //initial data at host side
    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;
    printf("initial Time elapsed:%f\n", iElaps);


    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nElem);

    //malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    //invoke kernel at host side
    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((nElem + block.x-1)/block.x);

    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f" \
        "sec\n", grid.x, block.x, iElaps);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    //copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    //add vector at host side for result checks
    iStart = cpuSecond();
    sumArraysCPU(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;
    printf("cpu add Elaps:%f\n", iElaps);

    cudaDeviceSynchronize();

    checkResult(hostRef, gpuRef, nElem);

    //free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);

    return 0;
}