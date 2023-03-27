#include <time.h>
#include <sys/time.h>
#include <stdio.h>

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

void transposeHost(float* hostRef, float* data, int nx, int ny){
    for(int iy=0;iy<ny;++iy){
        for(int ix=0;ix<nx;++ix){
            hostRef[ix*ny+iy] = data[iy*nx+ix];
        }
    }
}

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void copyRow(float *out, float *in, int nx, int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[iy*nx+ix] = in[iy*nx+ix];
    }
}

__global__ void copyCol(float *out, float *in, int nx, int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[ix*ny+iy] = in[ix*ny+iy];
    }
}

__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[ix*ny + iy] = in[iy*nx + ix]; // load by row write by col
    }
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx && iy<ny){
        out[iy*nx + ix] = in[ix*ny + iy]; // load by row write by col
    }
}

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x*4+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int ti = iy*nx+ix;
    unsigned int to = ix*ny+iy;

    if(ix+3*blockDim.x<nx && iy<ny){
        out[to] = in[ti];
        out[to+ny*blockDim.x] = in[ti+blockDim.x];
        out[to+ny*2*blockDim.x] = in[ti+2*blockDim.x];
        out[to+ny*3*blockDim.x] = in[ti+3*blockDim.x];

    }
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x*4+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    unsigned int ti = iy*nx+ix;
    unsigned int to = ix*ny+iy;

    if(ix+3*blockDim.x<nx && iy<ny){
        out[ti] = in[to];
        out[ti+blockDim.x] = in[to+blockDim.x*ny];
        out[ti+2*blockDim.x] = in[to+2*blockDim.x*ny];
        out[ti+3*blockDim.x] = in[to+3*blockDim.x*ny];

    }
}

__global__ void transposeDiagonalRow(float *out, float *in, int nx, int ny){
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x+blockIdx.y)%gridDim.x;

    unsigned int ix = blockDim.x*blk_x+threadIdx.x;
    unsigned int iy = blockDim.x*blk_y+threadIdx.y;

    if(ix<nx&&iy<ny){
        out[ix*ny+iy] = in[iy*nx+ix];
    }
}

__global__ void transposeDiagonalCol(float *out, float *in, int nx, int ny){
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x+blockIdx.y)%gridDim.x;

    unsigned int ix = blockDim.x*blk_x+threadIdx.x;
    unsigned int iy = blockDim.x*blk_y+threadIdx.y;

    if(ix<nx&&iy<ny){
        out[iy*nx+ix] = in[ix*ny+iy];
    }
}


__global__ void warmup(){
    unsigned int ix = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int iy = blockDim.y*blockIdx.y+threadIdx.y;
    if(ix == 1 && iy ==1) printf("test\n");
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
    int blockx = 16;
    int blocky = 16;
    if(argc>1) iKernel = atoi(argv[1]);
    if(argc>2) blockx  = atoi(argv[2]);
    if(argc>3) blocky  = atoi(argv[3]);
    if(argc>4) nx      = atoi(argv[4]);
    if(argc>5) ny      = atoi(argv[5]);

    printf("with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    int nBytes = nx*ny*sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printf("nBytes:%d\n", nBytes);

    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nx*ny);

    transposeHost(hostRef, h_A, nx, ny);

    float *d_A, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    warmup<<<grid, block>>>();

    cudaMemcpy(d_A, h_A, nx*ny, cudaMemcpyHostToDevice);

    void (*kernel)(float *, float *, int , int);
    char *kernelName;

    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "copyRow";
        break;
    
    case 1:
        kernel = &copyCol;
        kernelName = "copyCol";
        break;

    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "transposeNaiveRow";
        break;
    
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "transposeNaiveCol";
        break;

    case 4:
        kernel = &transposeUnroll4Col;
        kernelName = "transposeUnroll4Col";
        grid.x = (nx+block.x+1)/(block.x*4);
        break;

    case 5:
        kernel = &transposeUnroll4Row;
        kernelName = "transposeUnroll4Row";
        grid.x = (nx+block.x+1)/(block.x*4);
        break;

    case 6:
        kernel = &transposeDiagonalRow;
        kernelName = "transposeDiagonalRow";
        break;
    
    case 7:
        kernel = &transposeDiagonalCol;
        kernelName = "transposeDiagonalCol";
        break;
    }

    double iStart = seconds();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;

    float ibnd = 2*nx*ny*sizeof(float)/1e9/iElaps;
    printf("%s elpased %f sec <<< grid (%d,%d) block (%d,%d) >>> "
    "effective bandwidth %f GB\n", kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);

    if(iKernel>1){
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
        checkResult(hostRef, gpuRef, nx*ny);
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;

}