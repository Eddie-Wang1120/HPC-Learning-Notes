#include <omp.h>
#include <stdio.h>
#include <unistd.h>
 
int main()
{
#pragma omp parallel num_threads(4) default(none)
   {
// sections 后面存在隐藏同步点
#pragma omp sections nowait
      {
#pragma omp section
         {
            int s = omp_get_thread_num() + 1;
            sleep(s);
            printf("tid = %d sleep %d seconds\n", s, s);
         }
#pragma omp section
         {
            int s = omp_get_thread_num() + 1;
            sleep(s);
            printf("tid = %d sleep %d seconds\n", s, s);
         }
#pragma omp section
         {
            int s = omp_get_thread_num() + 1;
            sleep(s);
            printf("tid = %d sleep %d seconds\n", s, s);
         }
#pragma omp section
         {
            int s = omp_get_thread_num() + 1;
            sleep(s);
            printf("tid = %d sleep %d seconds\n", s, s);
         }
      }
 
      printf("tid = %d finish sections\n", omp_get_thread_num());
   }
   return 0;
}