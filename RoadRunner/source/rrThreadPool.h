#ifndef rrThreadPoolH
#define rrThreadPoolH
#include <list>
#include <vector>
#include "Poco/Thread.h"
#include "Poco/Mutex.h"
#include "Poco/ScopedLock.h"
#include "rrRoadRunnerThread.h"

//---------------------------------------------------------------------------
using namespace std;
using namespace rr;

namespace rr
{

class RR_DECLSPEC ThreadPool
{
	protected:
		list<RoadRunnerThread*>		mThreads;

    public:
								   	ThreadPool();
		void						addJob(RoadRunner* rri);
		bool						isJobQueueEmpty();
		int							getNumberOfRemainingJobs();
		bool						isWorking();
		void						start();
		void						exitAll();
		void						waitForStart();
		void						waitForFinish();

};

}
#endif
