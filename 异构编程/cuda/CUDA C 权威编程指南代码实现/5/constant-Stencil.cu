#include <stdio.h>
#define RADIUS 4
#define BDIM 4

void initialData(float* data, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        data[i] = (float)( rand() & 0xFF)/10.0f;
    }
}


__constant__ float coef[RADIUS + 1];   

__global__ void stencil_1d(float *in, float *out){
    __shared__ float smem[BDIM+2*RADIUS];
    //global index
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    //smem index
    int sidx = threadIdx.x + RADIUS;
    smem[sidx] = in[idx];
    //fill in the blank smem(<global >global)
    if(threadIdx.x<RADIUS){
        smem[sidx-RADIUS] = in[idx-RADIUS];
        smem[sidx+BDIM] = in[idx+BDIM];
    }

    __syncthreads();

    float tmp = 0.0f;
    #pragma unroll
    for(int i = 1;i<=RADIUS;i++){
        tmp += coef[i]*(smem[sidx+i]-smem[sidx-i]);
    }

    out[idx] = tmp;
}

void setup_coef_constant(void){
    float a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4;
    const float h_coef[] = {a0, a1, a2, a3, a4};
    cudaMemcpyToSymbol(coef, h_coef, (RADIUS+1)*sizeof(float));
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    printf("%s starting\n",deviceProp.name);

    setup_coef_constant();

    int nElem = BDIM;
    size_t nBytes = nElem*sizeof(nElem);

    float *h_A = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);

    float *d_A, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    dim3 block(BDIM, 1);
    dim3 grid(1, 1);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    stencil_1d<<<grid, block>>>(d_A, d_C);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i<BDIM;i++){
        printf("%f ",gpuRef[i]);
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(gpuRef);

    return EXIT_SUCCESS;

}