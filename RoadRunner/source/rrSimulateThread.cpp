#pragma hdrstop
#include "rrSimulateThread.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

namespace rr
{

int 				SimulateThread::mNrOfWorkers = 0;
Poco::Mutex 		SimulateThread::mNrOfWorkersMutex;

list<RoadRunner*>   SimulateThread::mJobs;
Mutex 				SimulateThread::mJobsMutex;
Condition			SimulateThread::mJobsCondition;

SimulateThread::SimulateThread(RoadRunner* rri, bool autoStart)
:
RoadRunnerThread()
{
	if(rri)
    {
    	addJob(rri);
    }

	if(autoStart && rri != NULL)
    {
    	start();
    }
}

void SimulateThread::addJob(RoadRunner* rr)
{
	//getMutex
    Mutex::ScopedLock lock(mJobsMutex);
    mJobs.push_back(rr);
	mJobsCondition.signal();	//Tell the thread its time to go to work
}

bool SimulateThread::isAnyWorking()
{
	bool result = false;
   	Mutex::ScopedLock lock(mNrOfWorkersMutex);
    return mNrOfWorkers > 0 ? true : false;
}

bool SimulateThread::isWorking()
{
	return mIsWorking;
}

void SimulateThread::worker()
{
    mWasStarted = true;
	mIsWorking  = true;

   	Mutex::ScopedLock lock(mNrOfWorkersMutex);
    mNrOfWorkers++;
    mNrOfWorkersMutex.unlock();

    RoadRunner *rri = NULL;
	//////////////////////////////////
    while(!mIsTimeToDie)
    {
        {	//Scope for the mutex lock...
            Mutex::ScopedLock lock(mJobsMutex);
            if(mJobs.size() == 0 )//|| mIsTimeToDie)
            {
                break;	//ends the life of the thread..
            }
                rri = mJobs.front();
                mJobs.pop_front();
         }		//Causes the scoped lock to unlock

        //Do the job
        if(rri)
        {
            Log(lInfo)<<"Simulating RR instance: "<<rri->getInstanceID();
            if(!rri->simulate2())
            {
                Log(lError)<<"Failed simulating instance: "<<rri->getInstanceID();
            }
        }
        else
        {
        	Log(lError)<<"Null job pointer...!";
        }
    }

    Log(lDebug)<<"Exiting Simulate thread: "<<mThread.id();

  	mIsWorking  = false;
   	Mutex::ScopedLock lock2(mNrOfWorkersMutex);
    mNrOfWorkers--;
}

void SimulateThread::signalExit()
{
	mJobsCondition.signal();
}

unsigned int SimulateThread::getNrOfJobsInQueue()
{
    Mutex::ScopedLock lock(mJobsMutex);
    return mJobs.size();
}

void SimulateThread::signalAll()
{
	mJobsCondition.broadcast();
}



}
