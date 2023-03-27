#include <stdio.h>
#include <omp.h>
#include <unistd.h>
 
int main()
{
#pragma omp parallel num_threads(4) default(none)
  {
//master线程为线程号为0的线程，无隐藏同步点
#pragma omp master
    {
      sleep(1);
      printf("In master construct tid = %d timestamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
    }
    printf("Out master construct tid = %d timestamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
  }
  return 0;
}