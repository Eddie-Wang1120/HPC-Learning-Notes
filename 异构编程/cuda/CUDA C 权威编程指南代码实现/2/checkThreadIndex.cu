#include <cstdio>

void initialInt(int *ip, int size){
    for(int i=0;i<size;i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix:%d, %d", nx, ny);
    for(int iy=0;iy<ny;iy++){
        for(int ix=0;ix<nx;ix++){
            printf("%3d\n",ic[ix]);
        }
        ic+=nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printGPUIdx(int* A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int idx = iy*nx + ix;

    printf("thread_id (%d,%d) block_id (%d %d) coordinate (%d %d) global index (%d) ival (%d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,ix, iy, idx, A[idx]);
}

void sumMatrixOnCPU(float *A, float *B, float*C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for(int iy=0;iy<ny;iy++){
        for(int ix=0;ix<nx;ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ic+=nx; ib+=nx; ia+=nx;
    }
}

__global__ void sumMatrixOnGPU(float *Mat_A, float *Mat_B, float *Mat_C, const int nx, const int ny){
    unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned idx = iy*nx + ix;

    if(ix < nx && iy < ny)
        Mat_C[idx] = Mat_A[idx] + Mat_B[idx];

}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy*(sizeof(float));

    //malloc host mem
    int *h_A;
    h_A = (int *)malloc(nBytes);

    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    //malloc device mem
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);

    //transfer
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    //set up excution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x-1)/block.x, (ny + block.y-1)/block.y);

    //invoke kernel
    printGPUIdx<<< grid, block >>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();

    //free host and device
    free(h_A);
    cudaFree(d_MatA);

    //reset device
    cudaDeviceReset();

    return 0;
}
