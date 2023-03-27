#include <cstdio>
#include <time.h>
#include <sys/time.h>

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

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        ip[i] = (float)( rand() & 0xFF)/10.0f;
    }
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

__global__ void sumMatrixOnGPU1D(float *Mat_A, float *Mat_B, float *Mat_C, const int nx, const int ny){
    unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
    if(ix<nx){
        for(int iy=0;iy<ny;iy++){
            int idx = iy*nx + ix;
            Mat_C[idx] = Mat_A[idx] + Mat_B[idx];
        }
    }
}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device:%d, %s\n", dev, deviceProp.name);

    //set up data for matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    int nBytes = nxy*(sizeof(float));
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    //malloc host mem
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    //initialize data at host side
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    printf("initial host matrix:%f\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnCPU(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("add matrix at host side:%f\n", iElaps);

    //malloc device global mem
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    //invoke cuda kernel
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x-1)/block.x, (ny + block.y-1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add matrix at device %f\n", iElaps);
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    //copy kernel result to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    //check device result
    checkResult(hostRef, gpuRef, nxy);
    
    //transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    dim3 block1D(32, 1);
    dim3 grid1D((nx+block1D.x-1)/block1D.x, 1);

    iStart = cpuSecond();
    sumMatrixOnGPU1D<<<block1D, grid1D>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add matrix at device %f\n", iElaps);
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>>\n", grid1D.x, grid1D.y, block1D.x, block1D.y);

    //check device result
    checkResult(hostRef, gpuRef, nxy);

    //free device global memory
    free(hostRef);
    free(gpuRef);
    free(h_A);
    free(h_B);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    //reset device
    cudaDeviceReset();

    return 0;
    

}