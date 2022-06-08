#pragma once

//platform-independent class for benchmarking
//TODO: reimplement as free functions in a namespace

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else

#include <sys/time.h>

#endif

class MyTime {
public:
    //define a type to store timestamps in
#ifdef _WIN32
    typedef LARGE_INTEGER Time_t;
#else
    typedef timeval Time_t;
#endif

public:

    //returns current time in system-dependant units
    inline static Time_t CTime() {
        Time_t tm;
#ifdef _WIN32
        QueryPerformanceCounter(&tm);
#else
        gettimeofday(&tm, NULL);
#endif
        return tm;
    }

    //time between two events, ms
    inline static float ElapsedTime(Time_t tmFirst, Time_t tmSecond) {
#ifdef _WIN32
        LARGE_INTEGER fq;
        QueryPerformanceFrequency(&fq);
        return (tmSecond.QuadPart-tmFirst.QuadPart)*1000.f/fq.QuadPart;
#else
        return ((float) (tmSecond.tv_sec - tmFirst.tv_sec) * 1000000
                + (tmSecond.tv_usec - tmFirst.tv_usec)) / 1000.f;
#endif
    }

private:
    MyTime(void);

    ~MyTime(void);
};

