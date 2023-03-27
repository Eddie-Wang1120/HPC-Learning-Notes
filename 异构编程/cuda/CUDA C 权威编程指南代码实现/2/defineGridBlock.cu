#include <cstdio>

int main(int argc, char **argv){
    int nElem  = 1024;

    dim3 block(1024);
    printf("%d\n", block.x);
    dim3 grid ((nElem + block.x-1)/block.x);
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    block.x = 512;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    block.x = 256;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    block.x = 128;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    cudaDeviceReset();
    return 0;
}