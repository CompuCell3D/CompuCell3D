#ifndef rrLoadModelThreadH
#define rrLoadModelThreadH
#include <list>
#include "rrExporter.h"
#include "rrRoadRunnerThread.h"
#include "Poco/Condition.h"

//---------------------------------------------------------------------------
using std::list;

namespace rr
{

class RR_DECLSPEC LoadModelThread : public RoadRunnerThread
{
	protected:
		string						mModelFileName;
    	static list<RoadRunner*>    mJobs;
		static Poco::Mutex	 		mJobsMutex;
        static Poco::Condition		mJobsCondition;

		static Poco::Mutex	 		mNrOfWorkersMutex;
        static int					mNrOfWorkers;		//incremented when entering worker function and decremented when exiting

        void						signalAll();
        void						signalExit();

	public:
    					            LoadModelThread(const string& modelFile, RoadRunner* rri = NULL, bool autoStart = false);
    	void 			            worker();
		void 			            addJob(RoadRunner* rr);
		unsigned int  				getNrOfJobsInQueue();
        bool	  					isAnyWorking();
};

}
#endif
