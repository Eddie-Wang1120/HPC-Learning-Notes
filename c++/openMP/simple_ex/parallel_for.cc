#include<omp.h>
#include<iostream>
using namespace std;
int main()
{
	omp_set_num_threads(2);//设置
    cout << omp_get_max_threads() << endl;
    cout << omp_get_num_threads() << endl;

    cout << "begin" << endl;

//schedule -> static(n/t) dynamic(先干完拿下一个) guided(大到小) runtime
#pragma omp parallel for schedule(dynamic)
		for(int i=0;i<4;i++)
		 cout << omp_get_thread_num() << endl;
}