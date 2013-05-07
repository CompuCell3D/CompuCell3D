#ifndef rrSimulateH
#define rrSimulateH
//---------------------------------------------------------------------------
#include "rrRoadRunnerList.h"
#include "rrThreadPool.h"

namespace rr
{

class RR_DECLSPEC Simulate : public ThreadPool
{
    public:
    	Simulate(RoadRunnerList& rrs, const int& nrThreads = 16);

};

}
#endif
