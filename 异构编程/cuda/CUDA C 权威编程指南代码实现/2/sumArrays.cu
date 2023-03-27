#include <cstdio>

__global__ void checkIndex(){
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z, gridDim.x,
        gridDim.y, gridDim.z);
}

int main(int argc, char **argv){
    int nElem = 6;
    dim3 block(3);
    dim3 grid ((nElem + block.x-1)/block.x);

    printf("grid.x %d grid.y %d, grid.z %d\n", grid.x, grid.y, grid.z);

    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}