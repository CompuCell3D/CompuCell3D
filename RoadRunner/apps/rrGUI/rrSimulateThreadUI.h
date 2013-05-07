#ifndef rrSimulateThreadUIH
#define rrSimulateThreadUIH
#include <list>
#include "rrRoadRunnerThread.h"
#include "Poco/Condition.h"
//---------------------------------------------------------------------------
class TMForm;

namespace rr
{

using std::list;
using Poco::Mutex;
using Poco::Condition;

class SimulateThreadUI : public RoadRunnerThread
{
	protected:
        static list<RoadRunner*>    mJobs;
		static Mutex	 			mJobsMutex;
    	static Condition			mJobsCondition;
        void						signalAll();
        void						signalExit();

		static Poco::Mutex	 		mNrOfWorkersMutex;
        static int					mNrOfWorkers;		//incremented when entering worker function and decremented when exiting

	public:
    								SimulateThreadUI(RoadRunner* rri = NULL, bool autoStart = false);
    	void 						worker();
		void 			            addJob(RoadRunner* rr);
		unsigned int  				getNrOfJobsInQueue();
        bool	  					isAnyWorking();
        bool	  					isWorking();

};

}
#endif
