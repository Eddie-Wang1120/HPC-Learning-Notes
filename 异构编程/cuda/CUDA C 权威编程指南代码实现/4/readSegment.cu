#include <stdio.h>
#include <time.h>
#include <sys/time.h>

//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 0
//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 11
//sudo nvprof --devices 0 --metrics gld_transactions ./readSegment 128

//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 0
//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 11
//sudo nvprof --devices 0 --metrics gld_efficiency ./readSegment 128


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

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void warmup(float* A, float* B, float* C, int nElem, int offset){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

__global__ void readOffset(float* A, float* B, float* C, int nElem, int offset){
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int k = idx+offset;
    if(k<nElem)C[idx] = A[k]+B[k];
}

__global__ void readOffsetUnroll4(float* A, float* B, float* C, const int n, int offset){
    unsigned i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned k = i + offset;
    if((k + 3*blockDim.x)<n){
        C[i] = A[k] + B[k];
        C[i+blockDim.x] = A[k+blockDim.x] + B[k+blockDim.x];
        C[i+2*blockDim.x] = A[k+2*blockDim.x] + B[k+2*blockDim.x];
        C[i+3*blockDim.x] = A[k+3*blockDim.x] + B[k+3*blockDim.x];
    }
}

void sumArrayOnHost(float* h_a, float* h_b, float* h_c, const size_t size, int offset){
    for(int i=offset, k=0;i<size;i++,k++){
        h_c[k] = h_a[i] + h_b[i];
    }
}


int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);

    int nElem = 1<<24;
    size_t nBytes = nElem * sizeof(float);

    int blocksize = 512;
    int offset = 0;
    if(argc>1) offset = atoi(argv[1]);
    if(argc>2) blocksize = atoi(argv[2]);

    dim3 block(blocksize, 1);
    dim3 grid((nElem+block.x-1)/block.x, 1);

    float* h_A = (float *)malloc(nBytes);
    float* h_B = (float *)malloc(nBytes);
    float* hostRef = (float *)malloc(nBytes);
    float* gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    sumArrayOnHost(h_A, h_B, hostRef, nElem, offset);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elpsed %f sec\n", grid.x, block.x, offset, iElaps);

    dim3 blockunroll(blocksize, 1);
    dim3 gridunroll(((nElem+block.x-1)/block.x)/4, 1);

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elpsed %f sec\n", grid.x, block.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    iStart = seconds();
    readOffset<<<gridunroll, blockunroll>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffsetunroll <<< %4d, %4d >>> offset %4d elpsed %f sec\n", gridunroll.x, blockunroll.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}