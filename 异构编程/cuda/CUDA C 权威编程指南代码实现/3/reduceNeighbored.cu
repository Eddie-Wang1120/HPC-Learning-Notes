
#include <cstdio>
#include <time.h>
#include <sys/time.h>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n ){
    //set thread ID
    unsigned int tid = threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //boundary check
    if(tid>=n) return;

    //in-place reduction in global memory
    for(int stride=1;stride<blockDim.x;stride*=2){
        if((tid%(2*stride)) == 0){
            idata[tid] += idata[tid+stride];
        }

        // //less time
        // for(int stride = 1;stride<blockDim.x;stride*=2){
        //     int index = 2*stride*tid;
        //     if(index<blockDim.x){
        //         idata[index]+=idata[index+stride];
        //     }
        // }

        //synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    //two block in one thread
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*2+threadIdx.x;

    int *idata = g_idata + blockIdx.x*blockDim.x*2;

    for(int stride = blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if(tid==0) g_odata[blockIdx.x] = idata[0];
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int recursiveReduce(int *data, int const size){
    if(size == 1) return data[0];
    int const stride = size/2;
    for(int i=0;i<stride;i++){
        data[i] += data[i+stride];
    }
    return recursiveReduce(data, stride);
}

int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    int size = 1<<24;
    printf("    with array size %d    ",size);

    int blockSize = 512;
    if(argc>1) blockSize = atoi(argv[1]);

    dim3 block(blockSize,1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("grid: %d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    size_t bytes = size*sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x*sizeof(int));
    int *tmp     = (int *)malloc(bytes);

    for(int i=0;i<size;i++){
        h_idata[i] = (int)(rand()&0xFF);
        }
    memcpy(tmp, h_idata, bytes);

    int iStart, iElaps;
    int gpu_sum=0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x*sizeof(int));

    iStart = cpuSecond();
    int cpu_sum =     recursiveReduce(tmp, size);
    iElaps = cpuSecond()-iStart;
    printf("cpu reduce  elapsed %d ms cpu_sum:%d\n", iElaps, cpu_sum);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum=0;
    for(int i=0;i<grid.x;i++){
        gpu_sum+=h_odata[i];
    }
    printf("gpu reduce  elapsed %d ms gpu_sum:%d\n", iElaps, gpu_sum);

    cudaMemAccessDesc

    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");
    else printf("Test succeed!\n");
    return 0;
}