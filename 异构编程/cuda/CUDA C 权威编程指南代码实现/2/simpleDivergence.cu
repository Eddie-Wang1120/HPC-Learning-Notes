#include <cstdio>
#include <time.h>
#include <sys/time.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void mathKernel1(float *c){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float a,b;
    a = b = 0.0f;
    if(tid % 2 ==0){
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float a,b;
    a = b = 0.0f;
    if((tid/warpSize) % 2 ==0){//避免线程束分化
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

int main(int argc, char **argv){
    //set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    //set up data size
    int size = 64;
    int blocksize = 64;
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size      = atoi(argv[2]);
    printf("data size %d \n", size);

    //set up excution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x -1)/block.x, 1);
    printf("Excution Configure (block %d grid %d)\n", block.x, grid.x);

    //allocate gpu memory
    float *d_C;
    int nBytes = size*(sizeof(float));
    cudaMalloc((float **)&d_C, nBytes);

    //run without warpSize
    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add in mathKernel1 needs time :%f\n", iElaps);

    //run with warpSize
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add in mathKernel2 needs time :%f\n", iElaps);

    return 0;


    //nvcc -O3 simpleDivergence.cu -o simpleDivergence
}