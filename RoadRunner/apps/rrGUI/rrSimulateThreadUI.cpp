#pragma hdrstop
#include "rrSimulateThreadUI.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

namespace rr
{

int 				SimulateThreadUI::mNrOfWorkers = 0;
Poco::Mutex 		SimulateThreadUI::mNrOfWorkersMutex;

list<RoadRunner*>   SimulateThreadUI::mJobs;
Mutex 				SimulateThreadUI::mJobsMutex;
Condition			SimulateThreadUI::mJobsCondition;

SimulateThreadUI::SimulateThreadUI(RoadRunner* rri, bool autoStart)
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

void SimulateThreadUI::addJob(RoadRunner* rr)
{
	//getMutex
    Mutex::ScopedLock lock(mJobsMutex);
    mJobs.push_back(rr);
	mJobsCondition.signal();	//Tell the thread its time to go to work
}

bool SimulateThreadUI::isAnyWorking()
{
	bool result = false;
   	Mutex::ScopedLock lock(mNrOfWorkersMutex);
    return mNrOfWorkers > 0 ? true : false;
}

bool SimulateThreadUI::isWorking()
{
	return mIsWorking;
}

void SimulateThreadUI::worker()
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

void SimulateThreadUI::signalExit()
{
	mJobsCondition.signal();
}

unsigned int SimulateThreadUI::getNrOfJobsInQueue()
{
    Mutex::ScopedLock lock(mJobsMutex);
    return mJobs.size();
}

void SimulateThreadUI::signalAll()
{
	mJobsCondition.broadcast();
}


}

//
//#ifdef USE_PCH
//#include "rr_pch.h"
//#endif
//#pragma hdrstop
//#include "rrLogger.h"
//#include "rrRoadRunner.h"
//#include "rrSimulateThreadUIUI.h"
//#include "rrException.h"
//#include "rrSimulationData.h"
//#include "MainForm.h"
////---------------------------------------------------------------------------
//#pragma package(smart_init)
//
//namespace rr
//{
//SimulateThreadUI::SimulateThreadUI(RoadRunner* rr, TMForm* mainForm)
//:
//mRR(rr),
//mHost(mainForm)
//{}
//
//SimulateThreadUI::~SimulateThreadUI()
//{}
//
//void SimulateThreadUI::AssignRRInstance(RoadRunner* rr)
//{
//	mRR = rr;
//}
//
//void SimulateThreadUI::Worker()
//{
//	mIsStarted = true;
//	mIsRunning = true;
//    int nrOfSimulations = 0;
//    while(!mIsTimeToDie)
//    {
//        if(mHost)
//        {
//        	if(!mHost->mData)
//            {
//                mHost->mData = new SimulationData;
//                try
//                {
//                	mRR->simulateEx(0, 1+ nrOfSimulations, 1000);
//                    nrOfSimulations++;
//                    *(mHost->mData) = mRR->getSimulationResult();
//                }
//                catch(const RRException& e)
//                {
//                    Log(lInfo)<<"Exception in RoadRunner Worker: "<<e.what();
//                }
//
//                TThread::Synchronize(NULL, mHost->PlotFromThread);
//		        Log(lInfo)<<"Simulation number: "<<nrOfSimulations;
//            }
//        }
//
//    }
//	Log(lInfo)<<"Simulate thread is exiting";
//	mIsRunning = false;
//    mIsFinished = true;
//}
//
//}
