#include <stdio.h>
#define LEN 1<<20

#include <time.h>
#include <sys/time.h>

struct innerStruct
{
    int x;
    int y;
};

// struct innerStruct myAoS[N];

//sudo nvprof --devices 0 --metrics gld_efficiency,gst_efficiency ./simpleMathAoS



void initialInnerStruct(innerStruct* A, size_t N){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<N;i++){
        A[i].x = (float)( rand() & 0xFF)/10.0f;
        A[i].y = (float)( rand() & 0xFF)/10.0f;
    }
}

void testInnerStructHost(innerStruct* A, innerStruct* B, size_t N){
    for(int i=0;i<N;i++){
        innerStruct tmp = A[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        B[i] = tmp;
    }
}

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}


double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef[i].x-gpuRef[i].x)>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
        if(abs(hostRef[i].y-gpuRef[i].y)>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if(match) printf("match!\n");
    return;

}

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);
    
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *) malloc(nBytes);

    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    innerStruct *d_A, *d_C;
    cudaMalloc((innerStruct**)&d_A, nBytes);
    cudaMalloc((innerStruct**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksize = 128;
    if(argc>1) blocksize = atoi(argv[1]);

    dim3 block (blocksize, 1);
    dim3 grid((nElem + block.x -1)/ block.x, 1);

    double iStart = seconds();
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("innerStruct <<< %4d, %4d >>> elpsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    free(h_A);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}
