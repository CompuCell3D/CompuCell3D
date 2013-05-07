#ifndef fit_one_parameterH
#define fit_one_parameterH
#include "rrCapability.h"
#include "rrParameter.h"
#include "rrPlugin.h"


//---------------------------------------------------------------------------

namespace fitOne
{
using namespace rr;

class FitOneParameter : public Plugin
{
	private:
    	Capability				mOneParameterFit;
    	Parameter<int>			mNrOfIterations;

    	Capability				mOneParameterFitResult;
   		Parameter<double>		mChiSquare;
        Parameter<double*>		mData;


    public:
    							FitOneParameter(rr::RoadRunner* aRR = NULL);
					   		   ~FitOneParameter();
		bool					execute();

};


extern "C"
{
#define EXP_FUNC __declspec(dllexport)
EXP_FUNC rr::Plugin* __stdcall	createRRPlugin(rr::RoadRunner* aRR);

}

}
#endif
