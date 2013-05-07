#ifndef rrSBMLModelSimulationH
#define rrSBMLModelSimulationH
//---------------------------------------------------------------------------
#include <string>
#include "rrObject.h"
#include "rrStringUtils.h"
#include "rrSimulationSettings.h"
#include "rrSimulationData.h"

namespace rr
{

class RoadRunner;
class RR_DECLSPEC SBMLModelSimulation : public rrObject
{
    protected:
        string                  mModelFileName;
        string                  mModelFilePath;
        string                  mModelSettingsFileName;
        string                  mSimulationLogFile;
        string                  mDataOutputFolder;
        string                  mTempDataFolder;
        RoadRunner             *mEngine;
        SimulationSettings      mSettings;
        bool                    mCompileIfDllExists;

    public:
                                SBMLModelSimulation(const string& dataOutputFolder = gEmptyString, const string& tempDataFilePath = gEmptyString);
        virtual                ~SBMLModelSimulation();
        bool                    SetModelFilePath(const string& path);
        bool                    SetModelFileName(const string& name);
        bool                    SetDataOutputFolder(const string& name);
        string                  GetModelsFullFilePath();
        string                  GetDataOutputFolder();
        string                  GetTempDataFolder();
        bool                    UseEngine(RoadRunner* engine);



        bool                    GenerateModelCode();
        bool                    CreateModel();
        bool                    CompileModel();
        bool                    InitializeModel();
        bool                    GenerateAndCompileModel();
        void                    ReCompileIfDllExists(const bool& doIt);
        bool                    DoCompileIfDllExists();
        bool                    SaveModelAsXML(const string& folder);

        //API
        bool                    SetTimeStart(const double& tStart);
        bool                    SetTimeEnd(const double& tEnd);
        bool                    SetNumberOfPoints(const int& pts);
        bool                    SetSelectionList(const string& list);
        virtual bool            LoadSBMLFromFile();                    //Use current file information to load sbml from file
        virtual bool            Simulate();
        virtual bool            SaveResult();
        virtual bool            LoadSettings(const string& fName = gEmptyString);
        virtual SimulationData  GetResult();
};

}



#endif
