//atomicCAS->原子级比较并交换符
//1 读取目标地址并将该地址存储值与预期值比较
//  相等-》新值存入目标位置
//  不等-》目标位置不变
//2 CAS总返回目标地址中值

__device__ int myAtomicAdd(int *address, int incr){
    int expected = *address;
    int oldValue = atomicCAS(address, expected, expected+incr);

    while(oldValue!=expected){
        expected = oldValue;
        oldValue = atomicCAS(address, expected, expected+incr);
    }

    return oldValue;
}