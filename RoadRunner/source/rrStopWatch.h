#ifndef rrStopWatchH
#define rrStopWatchH
//---------------------------------------------------------------------------
#include "rrObject.h"

namespace rr
{

class RR_DECLSPEC StopWatch : public rrObject
{
    private:
        int			 		mStartTime;			//Ticks...
        int			 		mTotalTime;
        bool 				mIsRunning;
        int					GetMilliSecondCount();
        int					GetMilliSecondSpan();

    public:
        					StopWatch();
        				   ~StopWatch();
        int 				Start();
        int 				Stop();
        double 				GetTime();
};
}

#endif
