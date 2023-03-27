#include <omp.h>
#include <iostream>

using namespace std;

int main(){
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        //有nowait则无隐式同步点
        #pragma omp single nowait
        {
            cout << "single thread=" << omp_get_thread_num() << endl;
        }
        cout << omp_get_thread_num() << endl;
    }
}