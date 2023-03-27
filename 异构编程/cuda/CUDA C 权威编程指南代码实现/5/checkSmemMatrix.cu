#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#define BDIMX 32
#define BDIMY 16


__global__ void setRowReadRow(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out){
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y*blockDim.x+threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y*blockDim.x+threadIdx.x;
    
    unsigned int irow = idx/blockDim.y;
    unsigned int icol = idx%blockDim.y;
    
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int *out){
    extern __shared__ int tile[];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx/blockDim.y;
    unsigned int icol = idx%blockDim.y;
    
    unsigned int col_idx = icol*blockDim.x+irow;

    tile[idx] = idx;
    __syncthreads();

    out[idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out){
    __shared__ int tile[BDIMY]{BDIMX+1};

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx/blockDim.y;
    unsigned int icol = idx%blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[icol][irow];
}

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    int iKernel = 0;

    if(argc>1) iKernel = atoi(argv[1]);

    int nElem = BDIMX*BDIMY;
    size_t nBytes = nElem*sizeof(int);

    int *gpuRef = (int *)malloc(nBytes);

    int *d_A;
    cudaMalloc((int **)&d_A, nBytes);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    void (*kernel)(int *);
    char *kernelName;

    switch (iKernel)
    {
    case 0:
        kernel = &setRowReadRow;
        kernelName = "setRowReadRow";
        break;
    
    case 1:
        kernel = &setColReadCol;
        kernelName = "setColReadCol";

    case 2:
        kernel = &setRowReadCol;
        kernelName = "setRowReadCol";

    case 3:
        kernel = &setRowReadColDyn;
        kernelName = "setRowReadColDyn";   

    case 4:
        kernel = &setRowReadColPad;
        kernelName = "setRowReadColPad";       
    }

    double iStart = seconds();
    kernel<<<grid, block>>>(d_A);
    double iElaps = seconds() - iStart;
    printf("%s elapsed %f sec\n",kernelName ,iElaps);

    cudaMemcpy(gpuRef, d_A, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i<34;i++){
        printf("%d ", gpuRef[i]);
    }
    
    cudaFree(d_A);
    cudaFree(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}