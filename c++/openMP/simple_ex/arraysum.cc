#include "timeclock.h"
#include <omp.h>
#include <iostream>

using namespace std;

int serial_arraysum(){
    int sum = 0;
    int a[10] = {1,2,3,4,5,6,7,8,9,10};
    for (int i=0;i<10;i++)
        sum = sum + a[i];
    return sum;
}

int parallel_worng_arraysum(){
    int sum = 0;
    int a[10] = {1,2,3,4,5,6,7,8,9,10};
#pragma omp parallel for
    for (int i=0;i<10;i++)
        sum = sum + a[i];
    return sum;
}

int parallel_right_arraysum(){
    int sum = 0;
    int a[10] = {1,2,3,4,5,6,7,8,9,10};
    int coreNum = omp_get_num_procs();//获得处理器个数
    int* sumArray = new int[coreNum];//对应处理器个数，先生成一个数组
    for (int i=0;i<coreNum;i++)//将数组各元素初始化为0
        sumArray[i] = 0;
#pragma omp parallel for
    for (int i=0;i<10;i++)
    {
        int k = omp_get_thread_num();//获得每个线程的ID
        sumArray[k] = sumArray[k]+a[i];
    }
    for (int i = 0;i<coreNum;i++)
        sum = sum + sumArray[i];
    return sum;
}

int parallel_reduction_arraysum(){
    int sum = 0;
    int a[10] = {1,2,3,4,5,6,7,8,9,10};
    // openmp归约指每个线程维护一个本身的sum变量，最后加到一起
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<10;i++){
        sum += a[i];
    }
    return sum;
}

int parallel_critical_arraysum(){
    int sum = 0;
    int a[10] = {1,2,3,4,5,6,7,8,9,10};
    #pragma omp parallel for
    for(int i=0;i<10;i++){
        #pragma omp critical //临界区
        {
            sum += a[i];
        }
    }
    return sum;
}


int main(){
    cout << parallel_critical_arraysum() << endl;
}