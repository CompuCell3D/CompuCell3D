#pragma hdrstop
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrRoadRunnerThread.h"

using namespace Poco;
namespace rr
{

RoadRunnerThread::RoadRunnerThread() :
mIsTimeToDie(false),
mWasStarted(false),
mIsWorking(false)
{}

bool RoadRunnerThread::wasStarted()
{
	return mWasStarted;
}
bool RoadRunnerThread::isWorking()
{
	return mIsWorking;
}

bool RoadRunnerThread::didFinish()
{
	return !isActive();
}

void RoadRunnerThread::setName(const string& name)
{
	mThread.setName(name);
}

string RoadRunnerThread::getName()
{
	return mThread.getName();
}

void RoadRunnerThread::exit()
{
	mIsTimeToDie = true;
    signalExit();
}

void RoadRunnerThread::start(RoadRunner* instance)
{
	if(instance)
    {
    	addJob(instance);
    }

	mIsTimeToDie = false;

    if(mIsWorking)
    {
    	Log(lError)<<"Tried to start a working thread!";
        return ;
    }

	mWasStarted = false;
	mIsWorking = false;

	mThread.start(*this);
	waitForStart();
}

void RoadRunnerThread::run()
{
	worker();
}

void RoadRunnerThread::join()
{
	mThread.join();
}

bool RoadRunnerThread::isActive()
{
	return mThread.isRunning();
}

void RoadRunnerThread::waitForStart()
{
    while(wasStarted() == false)
    {
        Poco::Thread::sleep(1);
    };
}

void RoadRunnerThread::waitForFinish()
{
    while(isActive() != false)
    {
        sleep(5);
    };
}

}