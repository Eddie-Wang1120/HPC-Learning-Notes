#include <stdio.h>

#define BDIMX 16
#define SEGM 4

//5.6.2.1 跨线程束值的广播
__global__ void test_shfl_broadcast(float *d_out, float *d_in, int const srcLane){
    float value = d_in[threadIdx.x];
    value = __shfl(value, srcLane, BDIMX);
    d_out[threadIdx.x] = value;
}

//5.6.2.2 线程束内上移
__global__ void test_shfl_up(float *d_out, float *d_in, unsigned int const delta){
    float value = d_in[threadIdx.x];
    value = __shfl_up(value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

//5.6.2.3 线程束下移
__global__ void test_shfl_down(float *d_out, float *d_in, unsigned int const delta){
    float value = d_in[threadIdx.x];
    value = __shfl_down(value, delta, BDIMX);
    d_out[threadIdx.x] = value;
}

//5.6.2.4 线程束内环绕移动
__global__ void test_shfl_warp(float *d_out, float *d_in, unsigned int const delta){
    float value = d_in[threadIdx.x];
    value = __shfl(value, threadIdx.x+offset, BDIMX);
    d_out[threadIdx.x] = value;
}

//5.6.2.5 垮线程束蝴蝶交换
__global__ void test_shfl_xor(float *d_out, float *d_in, int const mask){
    float value = d_in[threadIdx.x];
    value = __shfl_xor(value, mask, BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int *d_out, int *d_in, int const mask){
    //test_shfl_xor_int4<<1,BDIMX/SEGM>>>(d_out,d_in, 1);
    int idx = threadIdx.x*SEGM;
    int value[SEGM];

    for(int i=0;i<SEGM;i++) value[i] = d_in[idx+i];

    value[0] = __shfl_xor(value[0], mask, BDIMX);
    value[1] = __shfl_xor(value[1], mask, BDIMX);
    value[2] = __shfl_xor(value[2], mask, BDIMX);
    value[3] = __shfl_xor(value[3], mask, BDIMX);
    for(int i=0;i<SEGM;i++) d_out[idx+i] = value[i];
}

void initialData(float* data, int size){
    for(int i=0;i<size;i++){
        data[i] = i;
    }
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    printf("%s starting\n",deviceProp.name);


    int nElem = BDIMX;
    size_t nBytes = nElem*sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);

    float *d_A, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    dim3 block(BDIMX, 1);
    dim3 grid(1, 1);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    test_shfl_broadcast<<<grid, block>>>(d_C, d_A, 4);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i<BDIMX;i++){
        printf("%f ",gpuRef[i]);
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(gpuRef);

    return EXIT_SUCCESS;

}