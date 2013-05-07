#include <iostream>
#include <fstream>
#include "rrLogger.h"
#include "rrUtils.h"
#include "unit_test/UnitTest++.h"
#include "unit_test/XmlTestReporter.h"
#include "unit_test/TestReporterStdout.h"
#include "Args.h"
#include "rrRoadRunner.h"
#include "rrGetOptions.h"

using namespace std;
using namespace rr;
using namespace UnitTest;

string 	gRRInstallFolder 		= "";
string 	gSBMLModelsPath 		= "";
string  gTestDataFolder			= "";
string 	gCompiler 				= "";
string 	gSupportCodeFolder 		= "";
string 	gTempFolder		   		= "";
bool	gDebug			    	= false;

// initialized based on gSBMLModelsPath
string gTSModelsPath;

vector<string> gModels;
void ProcessCommandLineArguments(int argc, char* argv[], Args& args);

RoadRunner* gRR = NULL;

//call with arguments, -m"modelFilePath" -r"resultFileFolder" -t"TempFolder"
int main(int argc, char* argv[])
{
	clog<<"Entering cxx test application..\n";

    Args args;
    ProcessCommandLineArguments(argc, argv, args);


#if defined (WIN32)
gCompiler = "compilers\\tcc\\tcc.exe";
#else
gCompiler = "gcc";
#endif

	string reportFile(args.ResultOutputFile);

    gSBMLModelsPath 	= args.SBMLModelsFilePath;
    gTSModelsPath 		= JoinPath(JoinPath(gSBMLModelsPath, "cases"), "semantic");

    gTempFolder			= args.TempDataFolder;

  	string thisExeFolder = getCurrentExeFolder();
    Log(lDebug)<<"RoadRunner bin location is: "<<thisExeFolder;

    //Assume(!) this is the bin folder of roadrunner install
	gRRInstallFolder = getParentFolder(thisExeFolder);	//Go up one folder

    gCompiler	 		= JoinPath(gRRInstallFolder, gCompiler);
	gSupportCodeFolder 	= JoinPath(gRRInstallFolder, "rr_support");
	gTestDataFolder     = JoinPath(gRRInstallFolder, "tests");
   	bool doLogging  	= args.EnableLogging;
    if(doLogging)
    {
    	string logFile = JoinPath(gTempFolder, "RoadRunner.log");
        gLog.Init("", lDebug5);//, unique_ptr<LogFile>(new LogFile(logFile)));
        LogOutput::mLogToConsole = true;
    }

 	fstream aFile(reportFile.c_str(), ios::out);
    if(!aFile)
    {
		cerr<<"Failed opening report file: "<<reportFile<<" in rr_cpp_api testing executable.\n";
    	return -1;
    }

	XmlTestReporter reporter(aFile);
	TestRunner runner1(reporter);
//    TestReporterStdout reporter;
//	  TestRunner runner1(reporter);


//    clog<<"Running Base\n";
//    runner1.RunTestsIf(Test::GetTestList(), "Base", 			True(), 0);
//
//    clog<<"Running SteadyState Tests\n";
//    runner1.RunTestsIf(Test::GetTestList(), "SteadyState", 		True(), 0);
//
//    clog<<"Running ssThreeSpecies Tests\n";
//    runner1.RunTestsIf(Test::GetTestList(), "ssThreeSpecies", 	True(), 0);

//    clog<<"Running Stoichiometric Tests\n";
//    runner1.RunTestsIf(Test::GetTestList(), "Stoichiometric",   	True(), 0);

//    clog<<"Running TestSuite Tests\n";
    runner1.RunTestsIf(Test::GetTestList(), "SBML_l2v4",   	True(), 0);

    //Finish outputs result to xml file
    runner1.Finish();
//	Pause();
    return 0;
}

void ProcessCommandLineArguments(int argc, char* argv[], Args& args)
{
    char c;
    while ((c = GetOptions(argc, argv, ("vi:d:m:l:r:s:t:"))) != -1)
    {
        switch (c)
        {
            case ('m'): args.SBMLModelsFilePath                     = rrOptArg;                       break;
			//case ('l'): args.Compiler      			                = rrOptArg;                       break;
            case ('r'): args.ResultOutputFile                       = rrOptArg;                       break;
			//case ('s'): args.SupportCodeFolder     		            = rrOptArg;                       break;
			case ('t'): args.TempDataFolder        		            = rrOptArg;                       break;
			//case ('d'): args.DataOutputFolder      		            = rrOptArg;                       break;
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
//#pragma comment(lib, "roadrunner-static.lib")
#pragma comment(lib, "roadrunner.lib")
#pragma comment(lib, "unit_test-static.lib")

//If we compile using a shared roadrunner, link with these
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "rr-libstruct-static.lib")
#endif

