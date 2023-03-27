#include <stdio.h>
#define DIM 64

void checkResult(int *hostRef, int *gpuRef, const int N){
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

void initialData(int* data, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        data[i] = (float)( rand() & 0xFF)/10.0f;
    }

}

__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x*blockIdx.x;

    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= n) return;

    if(blockDim.x>=1024 && tid<512) idata[tid] += idata[tid+512];
    __syncthreads();
    if(blockDim.x>=512  && tid<256) idata[tid] += idata[tid+256];
    __syncthreads();
    if(blockDim.x>=256  && tid<128) idata[tid] += idata[tid+128];
    __syncthreads();
    if(blockDim.x>=64   && tid<64 ) idata[tid] += idata[tid+64 ];
    __syncthreads();

    if(tid<32){
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid+32];
        __syncthreads();
        vsmem[tid] += vsmem[tid+16];
        __syncthreads();
        vsmem[tid] += vsmem[tid+8 ];
        __syncthreads();
        vsmem[tid] += vsmem[tid+4 ];
        __syncthreads();
        vsmem[tid] += vsmem[tid+2 ];
        __syncthreads();
        vsmem[tid] += vsmem[tid+1 ];
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int smem[DIM];
    
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockDim.x*blockIdx.x;

    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= n) return;

    smem[tid] = idata[tid];

    if(blockDim.x>=1024 && tid<512) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x>=512  && tid<256) smem[tid] += smem[tid+256];
    __syncthreads();
    if(blockDim.x>=256  && tid<128) smem[tid] += smem[tid+128];
    __syncthreads();
    if(blockDim.x>=64   && tid<64 ) smem[tid] += smem[tid+64 ];
    __syncthreads();

    if(tid<32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8 ];
        vsmem[tid] += vsmem[tid+4 ];
        vsmem[tid] += vsmem[tid+2 ];
        vsmem[tid] += vsmem[tid+1 ];
    }
    if(tid == 0) g_odata[blockIdx.x] = smem[0];

}

__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int smem[DIM];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*4+threadIdx.x;
    int tmpSum = 0;
    if(idx+3*blockDim.x<=n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx+blockDim.x];
        int a3 = g_idata[idx+2*blockDim.x];
        int a4 = g_idata[idx+3*blockDim.x];
        tmpSum = a1+a2+a3+a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();

    if(blockDim.x>=1024&&tid<512) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x>=512&&tid<256)  smem[tid] += smem[tid+256];
    __syncthreads();
    if(blockDim.x>=256&&tid<128)  smem[tid] += smem[tid+128];
    __syncthreads();
    if(blockDim.x>=128&&tid<64 )  smem[tid] += smem[tid+64 ];
    __syncthreads();

    if(tid<32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    if(tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int n){
    extern __shared__ int smem[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*4+threadIdx.x;
    int tmpSum = 0;
    if(idx+3*blockDim.x<=n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx+blockDim.x];
        int a3 = g_idata[idx+2*blockDim.x];
        int a4 = g_idata[idx+3*blockDim.x];
        tmpSum = a1+a2+a3+a4;
    }
    smem[tid] = tmpSum;
    __syncthreads();

    if(blockDim.x>=1024&&tid<512) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x>=512&&tid<256)  smem[tid] += smem[tid+256];
    __syncthreads();
    if(blockDim.x>=256&&tid<128)  smem[tid] += smem[tid+128];
    __syncthreads();
    if(blockDim.x>=128&&tid<64 )  smem[tid] += smem[tid+64 ];
    __syncthreads();

    if(tid<32){
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid+32];
        
        vsmem[tid] += vsmem[tid+16];
        vsmem[tid] += vsmem[tid+8];
        vsmem[tid] += vsmem[tid+4];
        vsmem[tid] += vsmem[tid+2];
        vsmem[tid] += vsmem[tid+1];
    }
    if(tid == 0) g_odata[blockIdx.x] = smem[0];
}

// __global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n){
//     __shared__ int smem[DIM];
// }

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, dev);
    cudaSetDevice(dev);
    printf("device %s strating...\n", deviceprop.name);

    int nElem = 1<<10;
    size_t nBytes = nElem*sizeof(int);

    int *h_A = (int *)malloc(nBytes);
    int *GmemRef = (int *)malloc(nBytes);
    int *SmemRef = (int *)malloc(nBytes);
    int *SUmemRef = (int *)malloc(nBytes);
    int *SUDmemRef = (int *)malloc(nBytes);
    
    int *d_A, *d_C, *d_D, *d_E, *d_F;
    cudaMalloc((int **)&d_A, nBytes);
    cudaMalloc((int **)&d_C, nBytes);
    cudaMalloc((int **)&d_D, nBytes);
    cudaMalloc((int **)&d_E, nBytes);
    cudaMalloc((int **)&d_F, nBytes);
    initialData(h_A, nElem);

    for(int i=0;i<32;i++){
        printf("%d ",h_A[i]);
         
    }
    printf("\n");   
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    
    dim3 block(DIM, 1);
    dim3 grid ((nElem + block.x -1)/block.x, 1);

    reduceGmem<<<grid, block>>>(d_A, d_C, nElem);
    cudaMemcpy(d_C, GmemRef, nBytes, cudaMemcpyDeviceToHost);

    reduceSmem<<<grid, block>>>(d_A, d_D, nElem);
    cudaMemcpy(d_D, SmemRef, nBytes, cudaMemcpyDeviceToHost);

    checkResult(GmemRef, SmemRef, nElem);

    reduceSmemUnroll<<<grid.x/4, block>>>(d_A, d_E, nElem);
    cudaMemcpy(d_E, SUmemRef, nBytes, cudaMemcpyDeviceToHost);

    checkResult(GmemRef, SUmemRef, nElem);

    reduceSmemUnrollDyn<<<grid.x/4, block, DIM*sizeof(int)>>>(d_A, d_F, nElem);
    cudaMemcpy(d_F, SUDmemRef, nBytes, cudaMemcpyDeviceToHost);

    checkResult(GmemRef, SUDmemRef, nElem);

    for(int i=0;i<32;i++){
        printf("%d ", GmemRef[i]);
           
    }
    printf("\n"); 

    cudaDeviceReset();
    return EXIT_SUCCESS;

}