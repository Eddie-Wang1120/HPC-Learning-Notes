#include <stdio.h>
#include <omp.h>
 
int main()
{
 
#pragma omp parallel num_threads(4) default(none)
  {
#pragma omp for ordered
    for(int i = 0; i < 8; ++i)
    {
#pragma omp ordered
      printf("i = %d ", i);
    }
  }
  return 0;
}