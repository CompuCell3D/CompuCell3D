#ifndef TestSuiteSimulationH
#define TestSuiteSimulationH
//---------------------------------------------------------------------------
#include "rrTestSuiteModelSimulation.h"
#include "rrc_api.h"

using namespace rr;
using namespace rrc;

class TestSuiteSimulation : public rr::TestSuiteModelSimulation
{
	private:
		RRHandle 			mRRHandle;
		RRResultHandle		mResultHandle;

    public:
    						TestSuiteSimulation(const string& dataOutputFolder = "", const string& modelFilePath = "", const string& modelFileName = "");
                           ~TestSuiteSimulation();
		void	  			UseHandle(RRHandle handle);
        bool            	LoadSBMLFromFile();                    //Use current file information to load sbml from file
        bool            	LoadSettings(const string& fName = "");
		bool 				Simulate();
        bool            	SaveResult();
        SimulationData  	GetResult();
};

//Utility function..
bool RunTest(const string& version, int caseNumber);
#endif
