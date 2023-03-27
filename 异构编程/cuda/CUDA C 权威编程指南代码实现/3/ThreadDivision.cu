__global__ void mathKernel1(float *c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float a,b;

    if(tid%2==0){//线程级编程，会导致线程束分化
                //CUDA编译器会自动优化
        a = 100.0f;
    }else{
        b = 200.0f;
    }

    c[tid] = a+b;
}

__global__ void mathKernel2(float *c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if((tid/warpSize)%2==0){//使分支粒度为线程束大小倍数
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    c[tid] = a+b;
}