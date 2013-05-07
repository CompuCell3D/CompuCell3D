#include <iostream>
#include "rrRoadRunner.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"

using namespace rr;

int main(int argc, char** argv)
{

	try
    {
        LogOutput::mLogToConsole = true;

		const char* rootPath = "..";

        gLog.SetCutOffLogLevel(lInfo);
		string tmpFolder = JoinPath(rootPath, "temp");

        const string modelFile = JoinPath(rootPath, "models", "test_1.xml");

#if defined(WIN32)
        const string modelLib  = JoinPath(rootPath, "temp", "test_1.dll");
#else
        const string modelLib  = JoinPath(rootPath, "temp", "test_1.so");
#endif
        

        //Load modelFiles..
        Log(lInfo)<<" ---------- LOADING/GENERATING MODEL: "<<modelFile;

		RoadRunner rr1(tmpFolder);
        if(!rr1.loadSBMLFromFile(modelFile))	//This will generate a model DLL
        {
            Log(lError)<<"Failed to create model DLL....";
            return -1;
        }

		ModelSharedLibrary lib;
		

		if(lib.load(modelLib))
		{
			Log(lInfo)<<"Shared lib loaded succesfully...";
		}
		else
		{
			Log(lInfo)<<"Shared lib was NOT loaded succesfully...";
		}
     

    }
    catch(const Exception& ex)
    {
    	Log(lError)<<"There was a  problem: "<<ex.getMessage();
    }

    //Pause(true);
    return 0;
}
#if defined(CG_IDE) || defined(MSVS)
#pragma comment(lib, "roadrunner-static.lib")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#endif
