#ifndef TestPluginH
#define TestPluginH
#include "rrPlugin.h"
#include "rrCapability.h"
#include "rrParameter.h"
//---------------------------------------------------------------------------

namespace TestPlugin
{
using namespace rr;

class TestPlugin : public Plugin
{
	private:
        Capability		mTestCapability;
        Parameter<int>	mTestParameter;

    public:
    					TestPlugin(rr::RoadRunner* aRR = NULL, int testParameter = 0);
					   ~TestPlugin();
		bool			execute();

};

/////////////////////////////////////////////////////////////////////////
// Plugin "factory" functions. This is for plugins that create a class
//If we are to create a plugin not in C++, then another set of API functions need to be
//exposed..

extern "C"
{
#define EXP_FUNC __declspec(dllexport)
EXP_FUNC rr::Plugin* __stdcall	createRRPlugin(rr::RoadRunner* aRR);

// Plugin cleanup function
EXP_FUNC bool		__stdcall	destroyRRPlugin(rr::Plugin *plugin);

}

}

#endif
