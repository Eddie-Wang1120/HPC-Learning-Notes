#include <iostream>
#define THREAD_PER_BLOCK 128
#include <time.h>
#include <sys/time.h>
//shared memory bank冲突

// nvprof
// 共享内存占用率：
// achieved_occupancy

// 全局内存读写：
// gld_throughput
// gld_efficiency
// gld_transactions
// gld_transactions_per_request

// 共享内存读写：
// shared_efficiency
// shared_load_throughput
// shared_load_transactions
// shared_load_transactions_per_request
// shared_store_throughput
// shared_store_transactions
// shared_store_transactions_per_request

double seconds(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void reduce(int *d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int tid = threadIdx.x;

    sdata[tid] = d_in[idx];
    __syncthreads();

    for(unsigned int i=1;i<blockDim.x;i*=2){
        if(tid%(2*i)==0){
            sdata[tid] = sdata[tid] + sdata[tid+i];
        }
        __syncthreads();
    }

    if(tid==0){
        d_out[blockIdx.x] = sdata[tid];
    }
}

__global__ void reduce1(int* d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = d_in[idx];
    __syncthreads();

    for(int i=1;i<blockDim.x;i*=2){
        int index = 2*i*tid;//乘2因为要累加
        if(index<blockDim.x){
            sdata[index] = sdata[index] + sdata[index+i]; 
        }
        __syncthreads();
    }

    if(tid==0) d_out[blockIdx.x] = sdata[tid];
}

__global__ void reduce2(int* d_in, int* d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = d_in[idx];
    __syncthreads();

    for(int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            sdata[tid] = sdata[tid] + sdata[tid+s];
        }   //解决bank冲突
        __syncthreads();
    }

    if(tid==0) d_out[blockIdx.x] = sdata[tid];

}

__global__ void reduce3(int *d_in,int *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();
    //闲置线程先做一次加法，防止线程利用率低下

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

__device__ void warpReduce(volatile int* cache, int tid){
    cache[tid] += cache[tid+16];
    __syncwarp();
    cache[tid] += cache[tid+8];
    __syncwarp();
    cache[tid] += cache[tid+4];
    __syncwarp();
    cache[tid] += cache[tid+2];
    __syncwarp();
    cache[tid] += cache[tid+1];
    __syncwarp();   
}

__global__ void reduce4(int *d_in,int *d_out){
    __shared__ int sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>=32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp cudaprop;
    cudaGetDeviceProperties(&cudaprop, dev);
    cudaSetDevice(dev);

    int nElem = THREAD_PER_BLOCK*4;
    int nSize = nElem * sizeof(int);

    int *host = (int *)malloc(nSize);
    for(int i=0;i<nElem;i++){
        host[i] = i;
        // printf("%d ", host[i]);
    }
    // printf("\n");

    int *res = (int *)malloc(nSize);

    int *d_in, *d_out;
    cudaMalloc((int **)&d_in, nSize);
    cudaMalloc((int **)&d_out, nSize);

    dim3 block(THREAD_PER_BLOCK);
    dim3 grid ((nElem + block.x -1)/block.x);

    double start = seconds();

    cudaMemcpy(d_in, host, nSize, cudaMemcpyHostToDevice);

    reduce4<<<grid, block>>>(d_in,d_out);

    cudaMemcpy(res, d_out, nSize, cudaMemcpyDeviceToHost);

    double eLpes = seconds() - start;
    printf("%f\n", eLpes);

    int arraySum = 0;
    for(int i=0;i<nElem;i++){
        // std::cout<<res[i]<<std::endl;
        // printf("%d ", res[i]);
        arraySum += res[i];
    }    
    std::cout<<arraySum<<std::endl;

}