#ifndef TimerClock_H_
#define TimerClock_H_

#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

class TimerClock{
public:
    TimerClock(){update();}
    ~TimerClock(){}

    void update(){_start = high_resolution_clock::now();}
    long long getTimerMicroSec(){
        return duration_cast<microseconds>(high_resolution_clock::now() - _start).count();
    }

private:
    time_point<high_resolution_clock> _start;
};

#endif