#include <iostream>
#include "timeclock.h"
void simple_serial()
{
    for(int i=0;i<8;i++){
    int a = 0;
    for (int j=0;j<100000000;j++)
        a++;}
}

void simple_parallel()
{
    #pragma omp parallel for
    for(int i=0;i<8;i++){
        int a = 0;
        for (int j=0;j<100000000;j++)
            a++;
    }
}


int main()
{
    TimerClock ck;
    ck.update();
    simple_parallel();
    std::cout<<"time: "<<ck.getTimerMicroSec()<<std::endl;
}