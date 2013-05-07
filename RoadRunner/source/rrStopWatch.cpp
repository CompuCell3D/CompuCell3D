#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <sys/timeb.h>
#include "rrLogger.h"
#include "rrStopWatch.h"
//---------------------------------------------------------------------------

namespace rr
{
StopWatch::StopWatch()
:
mIsRunning(false)
{
	Start();
}

StopWatch::~StopWatch()
{}

int StopWatch::Start()
{
	timeb tb;
	ftime(&tb);
	mStartTime = tb.millitm + (tb.time & 0xfffff) * 1000;
    mIsRunning = true;
    return mStartTime;
}

int StopWatch::GetMilliSecondCount()
{
	timeb tb;
	ftime(&tb);
	return tb.millitm + (tb.time & 0xfffff) * 1000;
}

int StopWatch::GetMilliSecondSpan()
{
	int nSpan = GetMilliSecondCount() - mStartTime;
	if(nSpan < 0)
    {
		nSpan += 0x100000 * 1000;
    }
	return nSpan;
}

int StopWatch::Stop()
{
    mTotalTime = GetMilliSecondSpan();
    mIsRunning = false;
    mStartTime = 0;
    return mTotalTime;
}

double StopWatch::GetTime()
{
    if(mIsRunning)
    {
        return GetMilliSecondSpan();
    }
    else
    {
        return mTotalTime;
    }
}
}
