//nvcc --ptx -o foo.ptx foo.cu

__global__ void intrinsic(float *ptr){
    *ptr = __powf(*ptr, 2.0f);
}

__global__ void standard(float *ptr){
    *ptr = powf(*ptr, 2.0f);
}