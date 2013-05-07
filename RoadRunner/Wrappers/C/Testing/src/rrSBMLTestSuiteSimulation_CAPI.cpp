#pragma hdrstop
#include "rrException.h"
#include "rrUtils.h"
#include "rrSimulationData.h"
#include "rrSBMLTestSuiteSimulation_CAPI.h"

using namespace rrc;

extern string gTempFolder;
extern string gTSModelsPath;
extern bool gDebug;
using namespace rr;
SimulationData convertCAPIResultData(RRResultHandle		resultsHandle);

SBMLTestSuiteSimulation_CAPI::SBMLTestSuiteSimulation_CAPI(const string& dataOutputFolder, const string& modelFilePath, const string& modelFileName)
:
rr::TestSuiteModelSimulation(dataOutputFolder, modelFilePath, modelFileName)
{
}

SBMLTestSuiteSimulation_CAPI::~SBMLTestSuiteSimulation_CAPI()
{}

void SBMLTestSuiteSimulation_CAPI::UseHandle(RRHandle handle)
{
	mRRHandle = handle;
    if(mRRHandle)
    {
    	this->UseEngine((RoadRunner*) mRRHandle);
    }
}


bool SBMLTestSuiteSimulation_CAPI::LoadSBMLFromFile()
{
	if(!mRRHandle)
    {
    	return false;
    }

    return loadSBMLFromFileE(mRRHandle, GetModelsFullFilePath().c_str(), true);
}

bool SBMLTestSuiteSimulation_CAPI::LoadSettings(const string& settingsFName)
{
    mModelSettingsFileName = (settingsFName);

    if(!mModelSettingsFileName.size())
    {
        mModelSettingsFileName = JoinPath(mModelFilePath, GetSettingsFileNameForCase(mCurrentCaseNumber));
    }
	SBMLModelSimulation::LoadSettings(mModelSettingsFileName);

	return loadSimulationSettings(mRRHandle, mModelSettingsFileName.c_str());
}

bool SBMLTestSuiteSimulation_CAPI::Simulate()
{
    if(!mRRHandle)
    {
        return false;
    }
	mResultHandle = simulate(mRRHandle);

    if(mResultHandle)
    {
		mResultData = convertCAPIResultData(mResultHandle);
    }
    return mResultHandle ? true : false;
}

SimulationData SBMLTestSuiteSimulation_CAPI::GetResult()
{
	return mResultData; //Not that pretty.
}

bool SBMLTestSuiteSimulation_CAPI::SaveResult()
{
    string resultFileName(JoinPath(mDataOutputFolder, "rrCAPI_" + mModelFileName));
    resultFileName = ChangeFileExtensionTo(resultFileName, ".csv");

    if(!mResultHandle)
    {
    	return false;
    }

    ofstream fs(resultFileName.c_str());
    fs << mResultData;
    fs.close();
    return true;
}

SimulationData convertCAPIResultData(RRResultHandle	result)
{
	SimulationData resultData;

	StringList colNames;
	//Copy column names
    for(int i = 0; i < result->CSize; i++)
    {
    	colNames.Add(result->ColumnHeaders[i]);
    }

	resultData.setColumnNames(colNames);

    //Then the data
    int index = 0;
    resultData.allocate(result->RSize, result->CSize);

    for(int j = 0; j < result->RSize; j++)
    {
        for(int i = 0; i < result->CSize; i++)
        {
            resultData(j,i) = result->Data[index++];
        }
    }

	return resultData;
}


bool RunTest(const string& version, int caseNumber)
{
	bool result(false);
 	RRHandle gRR;

    //Create instance..
    gRR = createRRInstanceE(gTempFolder.c_str());

    if(gDebug && gRR)
    {
	    enableLoggingToConsole();
        setLogLevel("Debug5");
    }
    else
    {
        setLogLevel("Error");
    }

	//Setup environment
    setTempFolder(gRR, gTempFolder.c_str());

    if(!gRR)
    {
    	return false;
    }

    try
    {
		clog<<"Running Test: "<<caseNumber<<endl;
        string dataOutputFolder(gTempFolder);
        string dummy;
        string logFileName;
        string settingsFileName;

        //Create a log file name
        CreateTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);

        //Create subfolder for data output
        dataOutputFolder = JoinPath(dataOutputFolder, GetTestSuiteSubFolderName(caseNumber));

        if(!CreateFolder(dataOutputFolder))
        {
			string msg("Failed creating output folder for data output: " + dataOutputFolder);
            throw(rr::Exception(msg));
        }

       	SBMLTestSuiteSimulation_CAPI simulation(dataOutputFolder);

		simulation.UseHandle(gRR);

        //Read SBML models.....
        string modelFilePath(gTSModelsPath);
        string modelFileName;

        simulation.SetCaseNumber(caseNumber);
        CreateTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml", modelFilePath, modelFileName, settingsFileName);

        //The following will load and compile and simulate the sbml model in the file
        simulation.SetModelFilePath(modelFilePath);
        simulation.SetModelFileName(modelFileName);
        simulation.ReCompileIfDllExists(true);
        simulation.CopyFilesToOutputFolder();
	    setTempFolder(gRR, simulation.GetDataOutputFolder().c_str());
        setComputeAndAssignConservationLaws(gRR, false);

        if(!simulation.LoadSBMLFromFile())
        {
            throw("Failed loading sbml from file");
        }

        //Then read settings file if it exists..
        string settingsOveride("");
        if(!simulation.LoadSettings(settingsOveride))
        {
            throw("Failed loading simulation settings");
        }

        //Then Simulate model
        if(!simulation.Simulate())
        {
            throw("Failed running simulation");
        }

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            throw("Failed saving result");
        }

        if(!simulation.LoadReferenceData())
        {
            throw("Failed Loading reference data");
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        simulation.SaveAllData();
        simulation.SaveModelAsXML(dataOutputFolder);
        if(!result)
        {
        	clog<<"\t\tTest failed..\n";
        }
        else
        {
			clog<<"\t\tTest passed..\n";
        }
  	}
    catch(rr::Exception& ex)
    {
        string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
    	return false;
    }
 	return result;
}
