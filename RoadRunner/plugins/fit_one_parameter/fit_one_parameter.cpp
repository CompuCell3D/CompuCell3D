#pragma hdrstop
#include "fit_one_parameter.h"
#include "rrRoadRunner.h"

//---------------------------------------------------------------------------
namespace fitOne
{
using namespace rr;

FitOneParameter::FitOneParameter(rr::RoadRunner* aRR)
:
Plugin("FitOneParameter", "No Category", aRR),
mOneParameterFit("OneParameterFit", "Run", "Runs a one parameter fit"),
mNrOfIterations("Number of Iterations", 	10,   	"Number of Iterations"),
mChiSquare("ChiSquare", 			   	0.123, 	"ChiSquare"),
mOneParameterFitResult("FitResult", "", ""),
mData("Some Data", NULL,"Result..")
{
	//Setup the plugins capabilities
    mOneParameterFit.add(&mNrOfIterations);
	mOneParameterFit.add(&mChiSquare);

    mOneParameterFitResult.add(&mData);

    mCapabilities.push_back(mOneParameterFit);
    mCapabilities.push_back(mOneParameterFitResult);
}

FitOneParameter::~FitOneParameter()
{}

bool FitOneParameter::execute()
{
	pLog()<<"Executing the FitOneParameter plugin";
    //Create a fitting thread, start it and then return..
	mRR->loadSBMLFromFile("r:\\models\\feedback.xml");
    for(int i = 0; i < mNrOfIterations.getValue(); i++)
    {
    	mRR->simulate();
    }

	return true;
}

// Plugin factory function
rr::Plugin* __stdcall createRRPlugin(rr::RoadRunner* aRR)
{
    //allocate a new object and return it
	return new FitOneParameter(aRR);
}

}

#if defined(CG_UI)
    #if defined(STATIC_BUILD)
    	#pragma comment(lib, "roadrunner-static.lib")
    #else
    	#pragma comment(lib, "roadrunner.lib")
    #endif
#endif


