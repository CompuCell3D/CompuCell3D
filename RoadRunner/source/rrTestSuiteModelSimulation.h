#ifndef rrSBMLTestSuiteModelSimulationH
#define rrSBMLTestSuiteModelSimulationH
//---------------------------------------------------------------------------
#include <string>
#include "rrSBMLModelSimulation.h"
#include "rrStringUtils.h"
#include "rrSimulationSettings.h"
#include "rrSimulationData.h"

namespace rr
{

class RR_DECLSPEC TestSuiteModelSimulation : public SBMLModelSimulation
{
    protected:
        int                     mCurrentCaseNumber;                     //If simulating test suite cases...
        string                  mModelSettingsFileName;
        SimulationData          mResultData;
        SimulationData          mReferenceData;
        SimulationData          mErrorData;
        string                  GetSettingsFileNameForCase(int sim_case);
        string                  GetReferenceDataFileNameForCase(int caseNr);
        int                     mNrOfFailingPoints;
        double                  mLargestError;

    public:
                                TestSuiteModelSimulation(const string& dataOutputFolder = gEmptyString, const string& modelFilePath = gEmptyString, const string& modelFileName = gEmptyString);
                               ~TestSuiteModelSimulation();
        void                    SetCaseNumber(int cNr);
        bool                    LoadReferenceData();
        bool                    CreateErrorData();
        bool                    SaveAllData();
        bool                    CopyFilesToOutputFolder();
        double                  LargestError();
        bool                    Pass();
        int                     NrOfFailingPoints();
        virtual bool            LoadSettings(const string& fName = gEmptyString);
};

}



#endif
