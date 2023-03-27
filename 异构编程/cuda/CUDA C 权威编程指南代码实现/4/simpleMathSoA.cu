#include <stdio.h>
#define LEN 1<<4

#include <time.h>
#include <sys/time.h>

struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initialinnerArray(innerArray* A, size_t N){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<N;i++){
        A->x[i] = (float)( rand() & 0xFF)/10.0f;
        A->y[i] = (float)( rand() & 0xFF)/10.0f;
    }
}

void testinnerArrayHost(innerArray* data, innerArray* result, size_t N){
    for(int i=0;i<N;i++){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

__global__ void testinnerArray(innerArray *data, innerArray *result, int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N){
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}


double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkinnerArray(innerArray *hostRef, innerArray *gpuRef, const int N){
    double eplison = 1.0E-5;
    int match = 1;
    for(int i=0;i<N;i++){
        if(abs(hostRef->x[i]-gpuRef->x[i])>eplison){
            match = 0;
            printf("do not match\n");
            break;
        }
        if(abs(hostRef->y[i]-gpuRef->y[i])>eplison){
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
    printf("%d\n",nElem);
    unsigned int nBytes = nElem * sizeof(innerArray);
    printf("%d\n",sizeof(innerArray));
    printf("%d\n",nBytes);
    innerArray *h_A = (innerArray *)malloc(nBytes);
    innerArray *hostRef = (innerArray *)malloc(nBytes);
    innerArray *gpuRef = (innerArray *) malloc(nBytes);

    initialinnerArray(h_A, nElem);
    testinnerArrayHost(h_A, hostRef, nElem);

    innerArray *d_A, *d_C;
    cudaMalloc((innerArray**)&d_A, nBytes);
    cudaMalloc((innerArray**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blocksize = 128;
    if(argc>1) blocksize = atoi(argv[1]);

    dim3 block (blocksize, 1);
    dim3 grid((nElem + block.x -1)/ block.x, 1);

    double iStart = seconds();
    testinnerArray<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("innerArray <<< %4d, %4d >>> elpsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkinnerArray(hostRef, gpuRef, nElem);

    free(h_A);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}
