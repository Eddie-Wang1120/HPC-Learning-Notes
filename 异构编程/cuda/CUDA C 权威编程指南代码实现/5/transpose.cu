#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#define BDIMX 32
#define BDIMY 16
#define IPAD 1
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

//性能下界
__global__ void naiveGmem(float *out, float *in, int nx, int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[ix*ny+iy] = in[iy*nx+ix];
    }
}

//性能上界（并未转置）
__global__ void copyGmem(float *out, float *in, int nx, int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[iy*nx+ix] = in[iy*nx+ix];
    }
}

__global__ void transposeSmem(float *out, float *in, const int nx, const int ny){
    __shared__ float tile[BDIMY][BDIMX];
    //original
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    //linear global memory index for original
    unsigned int ti = iy*nx+ix;
    //thread index in transposed block
    unsigned int bidx = threadIdx.y*blockDim.x+threadIdx.x;
    
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.y;
    //coordinate in transposed matrix
    ix = blockIdx.y*blockDim.y+icol;
    iy = blockIdx.x*blockDim.x+irow;

    //linear global memory index for transposed matrix
    unsigned int to = iy*ny+ix;

    if(ix<nx&&iy<ny){
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();
        out[to] = tile[icol][irow];
    }
}

__global__ void transposeSmemPad(float *out, float *in, const int nx, const int ny){
    __shared__ float tile[BDIMY][BDIMX+IPAD];
    
    //original
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    //linear global memory index for original
    unsigned int ti = iy*nx+ix;
    //thread index in transposed block
    unsigned int bidx = threadIdx.y*blockDim.x+threadIdx.x;
    //块内转置
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.y;
    //coordinate in transposed matrix
    //块外转置
    ix = blockIdx.y*blockDim.y+icol;
    iy = blockIdx.x*blockDim.x+irow;
    //linear global memory index for transposed matrix
    unsigned int to = iy*ny+ix;

    if(ix<nx&&iy<ny){
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();
        out[to] = tile[icol][irow];
    }
}

__global__ void transposeSmemUnrollPad(float *out, float *in, int nx, int ny){
    __shared__ float tile[BDIMY*(BDIMX*2+IPAD)];
    unsigned int ix = 2*blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;

    unsigned int ti = iy*nx + ix;

    unsigned int bidx = blockDim.x*threadIdx.y+threadIdx.x;
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.y;

    unsigned int ix2 = blockIdx.y*blockDim.y + icol;
    unsigned int iy2 = 2*blockIdx.x*blockDim.x+irow;

    unsigned int to = iy2*ny + ix2;

    if((ix+blockDim.x)<nx&&iy<ny){
        unsigned int row_idx = threadIdx.y*(blockDim.x*2+IPAD)+threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx+BDIMX] = in[ti+BDIMX];

        __syncthreads();

        unsigned int col_idx = icol*(blockDim.x*2+IPAD)+irow;
        out[to] = tile[col_idx];
        out[to+ny*BDIMX] = tile[col_idx+BDIMX];
    }
}

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp DeviceProp;
    cudaGetDeviceProperties(&DeviceProp, dev);

    printf("device %d: %s starting...\n", dev, DeviceProp.name);

    int nx = 1<<11;
    int ny = 1<<11;

    int iKernel = 0;
    int blockx = BDIMX;
    int blocky = BDIMY;
    if(argc>1) iKernel = atoi(argv[1]);
    if(argc>2) blockx  = atoi(argv[2]);
    if(argc>3) blocky  = atoi(argv[3]);
    if(argc>4) nx      = atoi(argv[4]);
    if(argc>5) ny      = atoi(argv[5]);

    printf("with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    int nBytes = nx*ny*sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printf("block <<<%d,%d>>>\n", block.x, block.y);
    printf("grid  <<<%d,%d>>>\n", grid.x, grid.y);

    float *h_A = (float *)malloc(nBytes);
    float *naiveGmemRef = (float *)malloc(nBytes);
    float *copyGmemRef = (float *)malloc(nBytes);
    float *transposeSmemRef = (float *)malloc(nBytes);
    float *transposeSmemPadRef = (float *)malloc(nBytes);
    float *transposeSmemUnrollPadRef = (float *)malloc(nBytes);

    initialData(h_A, nx*ny);

    float *d_A, *d_C, *d_D, *d_E, *d_F, *d_G;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, nBytes);
    cudaMalloc((float **)&d_D, nBytes);
    cudaMalloc((float **)&d_E, nBytes);
    cudaMalloc((float **)&d_F, nBytes);
    cudaMalloc((float **)&d_G, nBytes);

    cudaMemcpy(d_A, h_A, nx*ny, cudaMemcpyHostToDevice);

    naiveGmem<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(naiveGmemRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    copyGmem<<<grid, block>>>(d_D, d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(copyGmemRef, d_D, nBytes, cudaMemcpyDeviceToHost);

    transposeSmem<<<grid, block>>>(d_E, d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(transposeSmemRef, d_E, nBytes, cudaMemcpyDeviceToHost);
    checkResult(naiveGmemRef, transposeSmemRef, nx*ny);

    transposeSmemPad<<<grid, block>>>(d_F, d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(transposeSmemPadRef, d_F, nBytes, cudaMemcpyDeviceToHost);
    checkResult(naiveGmemRef, transposeSmemPadRef, nx*ny);

    transposeSmemUnrollPad<<<grid.x/2, block>>>(d_G, d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaMemcpy(transposeSmemUnrollPadRef, d_G, nBytes, cudaMemcpyDeviceToHost);
    checkResult(naiveGmemRef, transposeSmemUnrollPadRef, nx*ny);

    printf("transposeSmemRef\n");
    for(int i=0;i<10;i++){
        printf("%f ", transposeSmemRef[i]);
    }
    printf("\n");

    printf("transposeSmemPadRef\n");
    for(int i=0;i<10;i++){
        printf("%f ", transposeSmemPadRef[i]);
    }
    printf("\n");

    printf("transposeSmemUnrollPadRef\n");
    for(int i=0;i<10;i++){
        printf("%f ", transposeSmemUnrollPadRef[i]);
    }
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_F);
    cudaFree(d_G);
    free(h_A);
    free(naiveGmemRef);
    free(copyGmemRef);
    free(transposeSmemRef);
    free(transposeSmemPadRef);
    free(transposeSmemUnrollPadRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}