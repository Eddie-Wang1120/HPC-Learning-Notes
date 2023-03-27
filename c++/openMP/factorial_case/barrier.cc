#include <omp.h>
#include <iostream>

int factorial(int n){
    int s = 1;
    for(int i=1;i<=n;++i){
        s *= i;
    }
    return s;
}

int main(){
    int data[16];
    // default->none(不指定就不是shared) shared(相反)
    // shared->指定共享变量
    #pragma omp parallel num_threads(4) default(none) shared(data)
    {
        int id = omp_get_thread_num();
        data[id] = factorial(id+1);
        // barrier不共享
        #pragma omp barrier
        long sum = 0;
        #pragma omp single
        {
            for(int i = 0; i < 4 ;i++){
                sum += data[i];
            }
            printf("final value = %lf\n", (double) sum);
        }
    
    }
    return 0;
}