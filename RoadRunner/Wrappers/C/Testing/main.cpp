
#include <iostream>
#include <fstream>
#include "rrLogger.h"
#include "rrUtils.h"
#include "unit_test/UnitTest++.h"
#include "unit_test/XmlTestReporter.h"
#include "unit_test/TestReporterStdout.h"
#include "rrc_api.h"
#include "rrGetOptions.h"
#include "src/Args.h"

using namespace std;
using namespace rr;
using namespace rrc;
using namespace UnitTest;

string 	gTempFolder		   		= "";
string 	gRRInstallFolder 		= "";
string  gTestDataFolder			= "";
bool	gDebug			    	= false;

string 	gTSModelsPath;

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
	gRRInstallFolder 	= getParentFolder(thisExeFolder);
    gDebug				= args.EnableLogging;
    gTSModelsPath 		= args.ModelsFilePath;
    gTempFolder			= args.TempDataFolder;

	gTestDataFolder  	= JoinPath(gRRInstallFolder, "testing");


	setInstallFolder(gRRInstallFolder.c_str());

    if(gDebug)
    {
	    enableLoggingToConsole();
        setLogLevel("Debug5");
    }
    else
    {
      setLogLevel("Error");
    }
    // set test suite model path (read from cmd line)
    gTSModelsPath = JoinPath(JoinPath(gTSModelsPath, "cases"), "semantic");

	string reportFile(args.ResultOutputFile);
 	fstream aFile(reportFile.c_str(), ios::out);
    if(!aFile)
    {
    	cerr<<"Failed opening report file: "<<reportFile<<" in rrc_api testing executable.\n";
    	return -1;
    }

	XmlTestReporter reporter1(aFile);
	TestRunner runner1(reporter1);

	clog<<"Running Suite CORE_TESTS\n";
	runner1.RunTestsIf(Test::GetTestList(), "CORE_TESTS", 			True(), 0);

	clog<<"Running Suite TEST_MODEL_1\n";
	runner1.RunTestsIf(Test::GetTestList(), "TEST_MODEL_1", 			True(), 0);

////	clog<<"Running Suite CORE_EXCEPTIONS\n";
////	runner1.RunTestsIf(Test::GetTestList(), "CORE_EXCEPTIONS", 		True(), 0);

	clog<<"Running Suite SBML_l2v4\n";
    clog<<"ModelPath "<<gTSModelsPath;
    runner1.RunTestsIf(Test::GetTestList(), "SBML_l2v4", 	True(), 0);

    //Finish outputs result to xml file
    runner1.Finish();
    return 0;
}

void ProcessCommandLineArguments(int argc, char* argv[], Args& args)
{
    char c;
    while ((c = GetOptions(argc, argv, ("vi:d:m:l:r:s:t:"))) != -1)
    {
        switch (c)
        {
            case ('m'): args.ModelsFilePath      	               = rrOptArg;                       break;
            case ('r'): args.ResultOutputFile                       = rrOptArg;                       break;
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

#if defined(STATIC_RRC)
#pragma comment(lib, "rrc_api-static.lib")
#else
#pragma comment(lib, "rrc_api.lib")
#endif

#if defined(STATIC_RR)
	#pragma comment(lib, "roadrunner-static.lib")
#else
	#pragma comment(lib, "roadrunner.lib")
#endif

#pragma comment(lib, "sundials_cvode")
#pragma comment(lib, "sundials_nvecserial")
#pragma comment(lib, "libf2c")
#pragma comment(lib, "blas")
#pragma comment(lib, "lapack")
#pragma comment(lib, "nleq-static")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "poco_xml-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#pragma comment(lib, "unit_test-static.lib")
#endif


