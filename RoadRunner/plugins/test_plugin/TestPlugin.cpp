#pragma hdrstop
#include "TestPlugin.h"
#include "rrCapability.h"
#include "rrRoadRunner.h"
#include "rrParameter.h"

#if defined(CG_UI)
    #if defined(STATIC_BUILD)
    	#pragma comment(lib, "roadrunner-static.lib")
    #else
    	#pragma comment(lib, "roadrunner.lib")
    #endif
#endif
//---------------------------------------------------------------------------
namespace TestPlugin
{
TestPlugin::TestPlugin(rr::RoadRunner* aRR, int testParameter)
:
rr::Plugin("TestPlugin", "No Category", aRR),
mTestParameter("NrOfIterations", 123, "Hint for Nr of iterations"),
mTestCapability("A Plugin Capability", "RunMe", "Exposing the RunMe Function")

{
	mTestCapability.setup("TestPlugin", "SetTimeCourseSelectionList", "A function in a plugin");
    mTestCapability.add(&mTestParameter);
    mCapabilities.push_back(mTestCapability);
}

TestPlugin::~TestPlugin()
{}

bool TestPlugin::execute()
{
	cout<<"Executing plugin...\n";
	if(mRR)
    {
    	mRR->setTimeCourseSelectionList("S2, S1");
    }
	return true;
}

// Plugin factory function
rr::Plugin* __stdcall createRRPlugin(rr::RoadRunner* aRR)
{
    //allocate a new object and return it
	return new TestPlugin(aRR);
}

}

