#include <stdio.h>
#define N 30000

int n_streams = 4;

__global__ void kernel_1(){
    double sum = 0.0;
    for(int i=0;i<N;i++){
        sum = sum+tan(0.1)*tan(0.1);
    }
}

__global__ void kernel_2(){
    double sum = 0.0;
    for(int i=0;i<N;i++){
        sum = sum+tan(0.1)*tan(0.1);
    }
}

__global__ void kernel_3(){
    double sum = 0.0;
    for(int i=0;i<N;i++){
        sum = sum+tan(0.1)*tan(0.1);
    }
}

__global__ void kernel_4(){
    double sum = 0.0;
    for(int i=0;i<N;i++){
        sum = sum+tan(0.1)*tan(0.1);
    }
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);

    printf("device %s starting ...\n", deviceProp.name);

    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    for(int i=0;i<n_streams;i++){
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(1);
    dim3 grid(1);
    float elapsed_time;
    cudaEventRecord(start);
    for(int i = 0; i< n_streams; i++)
    {
    kernel_1<<<grid,block,0,streams[i]>>>();
    }
    for(int i = 0; i< n_streams; i++)
    {
    kernel_2<<<grid,block,0,streams[i]>>>();
    }
    for(int i = 0; i< n_streams; i++)
    {
    kernel_3<<<grid,block,0,streams[i]>>>();
    }
    for(int i = 0; i< n_streams; i++)
    {
    kernel_4<<<grid,block,0,streams[i]>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Measured time for parallel execution = %.3f ms\n",elapsed_time);

    for(int i=0;i<n_streams;i++){
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}