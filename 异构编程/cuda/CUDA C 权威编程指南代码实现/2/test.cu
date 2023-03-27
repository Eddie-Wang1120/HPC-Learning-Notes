#include <cstdio>

void CPUhello(){
    printf("hello from cpu\n");
}

__global__ void GPUhello(){
    if(threadIdx.x==4){
        printf("hello from gpu thread 5\n");
    }
}

int main(){
    CPUhello();
    GPUhello<<<1, 5>>>();
    cudaDeviceReset();
}