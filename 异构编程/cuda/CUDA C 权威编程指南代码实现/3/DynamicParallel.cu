#include <iostream>

//nvcc -rdc=true DynamicParallel.cu -o DynamicParallel -lcudadevrt

__global__ void nestedHelloWorld(int const iSize, int iDepth){
    int tid = threadIdx.x;
    printf("Recusion=%d:Hello World from thread %d block %d\n",iDepth,tid,blockIdx.x);
    //condition to stop recursive execution
    if(iSize == 1) return;

    //reduce block size to half
    int nthreads = iSize>>1;

    //thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0){
        nestedHelloWorld<<<2,nthreads>>>(nthreads, ++iDepth);//可调为1,nthreads等等
        printf("-------> nested excution depth: %d\n", iDepth);
    }
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize){
    //set thread ID
    unsigned int tid = threadIdx.x;

    //convert globa pointer to local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    //stop condition
    if(isize == 2 && tid == 0){
        g_odata[blockIdx.x] = idata[0]+idata[1];
        return;
    }

    //nested invocation
    int istride = isize>>1;
    if(istride>1 && tid<istride){
        // in place reduction
        idata[tid] += idata[tid + istride];
    }

    //sync at lock level
    __syncthreads();//unnecessary

    //nested invocation to generate child grids
    if(tid==0){
        gpuRecursiveReduce<<<1, istride>>>(idata,odata,istride);
        //sync all grids launched in this block
        cudaDeviceSynchronize();
    }

    //sync at block level again
    __syncthreads();
}

int main(int argc, int** argv){
    int dev = 0;
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, dev);
    cudaSetDevice(dev);

    int size = 1<<10;
    int blockSize = 8;

    dim3 block(blockSize, 1);
    dim3 grid((1, 1));

    //host
    size_t bytes = size*sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(bytes);

    //device
    int *d_idata;
    int *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, bytes);

    //host->device
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    nestedHelloWorld<<<grid, block>>>(8,0);
    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_idata);
    cudaFree(d_odata);

    free(h_idata);
    free(h_odata);
}