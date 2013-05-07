#pragma hdrstop
#include "rrLoadModelThread.h"
#include "rrLoadModel.h"

//---------------------------------------------------------------------------

namespace rr
{

LoadModel::LoadModel(RoadRunnerList& rrs, const string& model, const int& nrThreads)
:
ThreadPool()
{
    //create nrThreads that can load SBML models
    for(int i = 0; i < nrThreads; i++)
    {
        LoadModelThread* lmThread = new LoadModelThread(model);
        mThreads.push_back(lmThread);
    }

    //The following will add jobs
    for(int i = 0; i < rrs.count(); i++)
    {
        addJob(rrs[i]);
    }

    //This will start processing the jobs..
    if(!isJobQueueEmpty() && nrThreads > 0)
    {
		start();
    	waitForStart();	//Make sure it get started before moving on..
    }
}


}