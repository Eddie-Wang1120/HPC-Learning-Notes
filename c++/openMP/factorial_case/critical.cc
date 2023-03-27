#include <stdio.h>
#include <omp.h>
#include <unistd.h>
 
int main()
{
//默认critical共用同一把锁
//critical后面声明锁名使用不同锁
#pragma omp parallel num_threads(4) default(none)
   {
#pragma omp sections
      {
#pragma omp section
         {
#pragma omp critical(A)
            {
               printf("tid = %d time stamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
               sleep(2);
            }
         }
#pragma omp section
         {
#pragma omp critical
            {
               printf("tid = %d time stamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
               sleep(2);
            }
         }
 
#pragma omp section
         {
#pragma omp critical
            {
               printf("tid = %d time stamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
               sleep(2);
            }
         }
 
#pragma omp section
         {
#pragma omp critical
            {
               printf("tid = %d time stamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
               sleep(2);
            }
         }
 
#pragma omp section
         {
#pragma omp critical
            {
               printf("tid = %d time stamp = %lf\n", omp_get_thread_num(), omp_get_wtime());
               sleep(2);
            }
         }
      }
   }
   return 0;
}