#include <iostream>
#include <fstream>
#include "rrLogger.h"
#include "rrUtils.h"
#include "Args.h"
#include "rrc_api.h"
#include "rrGetOptions.h"
#include "TestSuiteSimulation.h"

using namespace std;
using namespace rr;
//using namespace rrc;

string 	gTempFolder		   		= "";
string 	gRRInstallFolder 		= "";
bool	gDebug			    	= false;
string 	gTSModelsPath			= "";
void ProcessCommandLineArguments(int argc, char* argv[], Args& args);

//call with arguments, -m"modelFilePath" -r"resultFileFolder" -t"TempFolder"
int main(int argc, char* argv[])
{
    enableLoggingToConsole();
    Args args;
    ProcessCommandLineArguments(argc, argv, args);

    string thisExeFolder = getCurrentExeFolder();
    clog<<"RoadRunner bin location is: "<<thisExeFolder<<endl;

    //Assume(!) this is the bin folder of roadrunner install
	gRRInstallFolder = getParentFolder(thisExeFolder);	//Go up one folder
    gDebug				= args.EnableLogging;
    gTSModelsPath 		= args.SBMLModelsFilePath;
    gTempFolder			= args.TempDataFolder;
	setInstallFolder(gRRInstallFolder.c_str());

    if(gDebug)
    {
	    enableLoggingToConsole();
        setLogLevel("Debug5");
    }
    else
    {
      setLogLevel("lInfo");
    }
    // set full model path (read from cmd line)
    gTSModelsPath = JoinPath(JoinPath(gTSModelsPath, "cases"), "semantic");
	Log(lInfo)<<"Testing model: "<<args.ModelNumber;

    RunTest("l2v4", args.ModelNumber);
    return 0;
}

void ProcessCommandLineArguments(int argc, char* argv[], Args& args)
{
    char c;
    while ((c = GetOptions(argc, argv, ("vi:m:t:"))) != -1)
    {
        switch (c)
        {
            case ('i'): args.ModelNumber                     		= ToInt(rrOptArg);                 break;
            case ('m'): args.SBMLModelsFilePath                     = rrOptArg;                       break;
			case ('t'): args.TempDataFolder        		            = rrOptArg;                       break;
			case ('v'): args.EnableLogging        		            = true;                       break;
            case ('?'): cout<<Usage(argv[0])<<endl;
            default:
            {
                string str = argv[rrOptInd-1];
                if(str != "-?")
                {
                    cout<<"*** Illegal option:\t"<<argv[rrOptInd-1]<<" ***\n\n";
                }
                exit(0);
            }
        }
    }

    //Check arguments, and choose to bail here if something is not right...
    if(argc < 2)
    {
        cout<<Usage(argv[0])<<endl;
       	rr::Pause();
        cout<<"\n";
        exit(0);
    }
}


#if defined(CG_IDE)

//#if defined(STATIC_RTL)
//#pragma comment(lib, "rrc_api-static.lib")
//#else
#pragma comment(lib, "rrc_api.lib")
//#endif

//#if defined(STATIC_RR)
	#pragma comment(lib, "roadrunner-static.lib")
//#else
//	#pragma comment(lib, "roadrunner.lib")
//#endif

#pragma comment(lib, "sundials_cvode")
#pragma comment(lib, "sundials_nvecserial")
#pragma comment(lib, "libf2c")
#pragma comment(lib, "blas")
#pragma comment(lib, "lapack")
#pragma comment(lib, "nleq-static")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#pragma comment(lib, "unit_test-static.lib")
#endif


