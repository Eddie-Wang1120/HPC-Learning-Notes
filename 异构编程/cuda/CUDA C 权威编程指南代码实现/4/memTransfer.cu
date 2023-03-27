#include <stdio.h>



int main(int argc, char **argv){
    //set up device
    int dev = 0;
    cudaSetDevice(dev);

    //memory size
    unsigned int isize = 1<<22;
    unsigned int nbytes = isize * sizeof(float);

    //get device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2ffMB\n", dev, deviceProp.name, isize, nbytes/(1024.0f*1024.0f));

    //allocate the host memory
    float *h_a = (float *)malloc(nbytes);

    //allocate the device memory
    float *d_a;
    cudaMalloc((void **)&d_a, nbytes);

    // initialize the host memory
    for(unsigned int i=0;i<isize;i++){
        h_a[i] = 0.5f;
    }

    //transfer data from the host to the device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

    //transfer data from device to the host
    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_a);
    free(h_a);

    float *h_aPinned = (float *)malloc(nbytes);
    cudaError_t status = cudaMallocHost((void **)&h_aPinned, nbytes);
    if(status != cudaSuccess){
        fprintf(stderr, "Reeor returned from pinned host memory\n");
        exit(1);
    }

    //reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;

    

}