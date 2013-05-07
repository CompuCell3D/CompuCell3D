/**
 * @file rrc_api.cpp
 * @brief roadRunner C API 2012
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/

#pragma hdrstop
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "rrParameter.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerList.h"
#include "rrLoadModel.h"
#include "rrLoadModelThread.h"
#include "rrSimulate.h"
#include "rrSimulateThread.h"
#include "rrCGenerator.h"
#include "rrLogger.h"
#include "rrException.h"
#include "rrUtils.h"
#include "rrCapability.h"
#include "rrPluginManager.h"
#include "rrPlugin.h"
#include "rrc_api.h" 			// Need to include this before the support header..
#include "rrc_support.h"   //Support functions, not exposed as api functions and or data

#if defined(_MSC_VER)
	#include <direct.h>
	#define getcwd _getcwd
	#define chdir  _chdir
#elif defined(__BORLANDC__)
  	#include <dir.h>
#else
#include <unistd.h>
#endif
//---------------------------------------------------------------------------

namespace rrc
{
using namespace std;
using namespace rr;

RRHandle rrCallConv createRRInstance()
{
	try
    {
    	string rrInstallFolder(getParentFolder(getRRCAPILocation()));

#if defined(_WIN32) || defined(WIN32)
            string compiler(JoinPath(rrInstallFolder,"compilers\\tcc\\tcc.exe"));
#elif defined(__linux)
            string compiler("gcc");
#else
            string compiler("gcc");
#endif
	//RoadRunner(const string& tempFolder, const string& supportCodeFolder, const string& compiler)
	        return new RoadRunner(GetUsersTempDataFolder(), JoinPath(rrInstallFolder, "rr_support"), compiler);
    }
	catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRHandle rrCallConv createRRInstanceE(const char* tempFolder)
{
	try
    {
    	char* text1 = getRRCAPILocation();
        string text2 = getParentFolder(text1);
    	string rrInstallFolder(text2);
        freeText(text1);

#if defined(_WIN32) || defined(WIN32)
            string compiler(JoinPath(rrInstallFolder, "compilers\\tcc\\tcc.exe"));
#elif defined(__linux)
            string compiler("gcc");
#else
            string compiler("gcc");
#endif
		if(tempFolder != NULL && !FileExists(tempFolder))
    	{
        	stringstream msg;
            msg<<"The temporary folder: "<<tempFolder<<" do not exist";
            Log(lError)<<msg.str();
    		throw(Exception(msg.str()));
        }
        else if(tempFolder)
        {
	        return new RoadRunner(tempFolder, JoinPath(rrInstallFolder, "rr_support"), compiler);
        }
        else
        {
	        return new RoadRunner(GetUsersTempDataFolder(), JoinPath(rrInstallFolder, "rr_support"), compiler);
        }
    }
	catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRInstanceListHandle rrCallConv createRRInstances(int count)
{
	try
    {    	
		string tempFolder = GetUsersTempDataFolder();

		RoadRunnerList* listHandle = new RoadRunnerList(count, tempFolder);

        //Create the C list structure
		RRInstanceListHandle rrList = new RRInstanceList;
        rrList->RRList = (void*) listHandle;
        rrList->Count = count;

        //Create 'count' handles
        rrList->Handle = new RRHandle[count];

        //Populate handles
        for(int i = 0; i < count; i++)
        {
        	rrList->Handle[i] = (*listHandle)[i];
        }
    	return rrList;
    }
	catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv freeRRInstances(RRInstanceListHandle rrList)
{
	try
    {
    	//Delete the C++ list
        RoadRunnerList* listHandle = (RoadRunnerList*) rrList->RRList;

		delete listHandle;

        //Free  C handles
        delete [] rrList->Handle;

        //Free the C list
        delete rrList;
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

char* rrCallConv getInstallFolder()
{
	if(!gInstallFolder)
	{
		gInstallFolder = new char[2048];
		strcpy(gInstallFolder, "/usr/local");
	}
	return gInstallFolder;
}

bool rrCallConv setInstallFolder(const char* folder)
{
	try
    {
		gInstallFolder = new char[2048];
	    return strcpy(gInstallFolder, folder) != NULL ? true : false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  		return false;
    }
}

char* rrCallConv getVersion(RRHandle handle)
{
	try
    {
   		RoadRunner* rri = castFrom(handle);
		return createText(rri->getVersion());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}


char* rrCallConv getRRCAPILocation()
{
#if defined(_WIN32) || defined(WIN32)
	char path[MAX_PATH];
    HINSTANCE handle = NULL;
    const char* dllName = "rrc_api";
    handle = GetModuleHandle(dllName);
    int nrChars = GetModuleFileNameA(handle, path, sizeof(path));
	if(nrChars != 0)
    {
	    string aPath = ExtractFilePath(path);
        char* text = createText(aPath);
		return text;
    }
    return NULL;
#else
	return createText(JoinPath(getInstallFolder(),"/lib"));
#endif
}

char* rrCallConv getCopyright()
{
	try
    {
    	RRHandle handle = createRRInstance();
        if(!handle)
        {
        	return NULL;
        }

   		RoadRunner* rri = castFrom(handle);
        char* text = createText(rri->getCopyright());
        freeRRInstance(handle);

        return text;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

char* rrCallConv getInfo(RRHandle handle)
{
	try
    {
   		RoadRunner* rri = castFrom(handle);
        char* text = NULL;
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
        }
        else
        {
            text = createText(rri->getInfo());
        }
        return text;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

char* rrCallConv getlibSBMLVersion(RRHandle handle)
{
	try
    {
   		RoadRunner* rri = castFrom(handle);
        char* text = NULL;
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
        }
        else
        {
            text = createText(rri->getlibSBMLVersion());
        }
        return text;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
	}
}

char* rrCallConv getCurrentSBML(RRHandle handle)
{
	try
    {
   		RoadRunner* rri = castFrom(handle);
        char* text = NULL;
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
        }
        else
        {
            text = createText(rri->writeSBML());
        }
        return text;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
	}
}

//Flags and Options
bool rrCallConv setComputeAndAssignConservationLaws(RRHandle handle, const bool OnOrOff)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
        rri->computeAndAssignConservationLaws(OnOrOff);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return false;
     }
}

bool rrCallConv setTempFolder(RRHandle handle, const char* folder)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
	    return rri->setTempFileFolder(folder);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  		return false;
    }
}

char* rrCallConv getTempFolder(RRHandle handle)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
	    return createText(rri->getTempFolder());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv setCompiler(RRHandle handle, const char* fName)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
    	if(!rri)
    	{
        	setError(ALLOCATE_API_ERROR_MSG);
        	return false;
    	}
		if(rri->getCompiler())
		{
			return rri->getCompiler()->setCompiler(fName);
		}
		else
		{
			return false;
		}
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

bool rrCallConv setCompilerLocation(RRHandle handle, const char* folder)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
    	if(!rri)
    	{
        	setError(ALLOCATE_API_ERROR_MSG);
        	return false;
    	}
		if(rri->getCompiler())
		{
			return rri->getCompiler()->setCompilerLocation(folder);
		}
		else
		{
			return false;
		}
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

char* rrCallConv getCompilerLocation(RRHandle handle)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

	    return createText(rri->getCompiler()->getCompilerLocation());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv setSupportCodeFolder(RRHandle handle,const char* folder)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
    	if(!rri)
    	{
        	setError(ALLOCATE_API_ERROR_MSG);
        	return false;
    	}
		if(rri->getCompiler())
		{
			return rri->getCompiler()->setSupportCodeFolder(folder);
		}
		else
		{
			return false;
		}
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

char* rrCallConv getSupportCodeFolder(RRHandle handle)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);


	    return createText(rri->getCompiler()->getSupportCodeFolder());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}


char* rrCallConv getWorkingDirectory()
{
	try
    {
	    return createText(rr::getCWD());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv loadSBMLFromFile(RRHandle _handle, const char* fileName)
{
	try
    {
        //Check first if file exists first
        if(!FileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

    	RoadRunner* rri = castFrom(_handle);
        if(!rri->loadSBMLFromFile(fileName))
        {
            setError("Failed to load SBML semantics");	//There are many ways loading a model can fail, look at logFile to know more
            return false;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv loadSBMLFromFileE(RRHandle _handle, const char* fileName, bool forceRecompile)
{
	try
    {
        //Check first if file exists first
        if(!FileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

    	RoadRunner* rri = castFrom(_handle);
        if(!rri->loadSBMLFromFile(fileName, forceRecompile))
        {
            setError("Failed to load SBML semantics");	//There are many ways loading a model can fail, look at logFile to know more
            return false;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

RRJobHandle rrCallConv loadSBMLFromFileJob(RRHandle rrHandle, const char* fileName)
{

	try
    {
        //Check if file exists first
        if(!FileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" do not exist";
            setError(msg.str());
            return NULL;
        }

        RoadRunner* rr = castFrom(rrHandle);
        LoadModelThread* loadThread = new LoadModelThread(fileName);

        if(!loadThread)
        {
            setError("Failed to create a LoadModel Thread");
        }
        loadThread->addJob(rr);
        loadThread->start();
        return loadThread;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRJobsHandle rrCallConv loadSBMLFromFileJobs(RRInstanceListHandle _handles, const char* fileName, int nrOfThreads)
{
	try
    {
        //Check if file exists first
        if(!FileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" do not exist";
            setError(msg.str());
            return NULL;
        }

        RoadRunnerList *rrs = getRRList(_handles);
        LoadModel* tp = new LoadModel(*rrs, fileName, nrOfThreads);

        if(!tp)
        {
            setError("Failed to create a LoadModel Thread Pool");
        }
        return tp;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

bool rrCallConv waitForJob(RRJobHandle handle)
{
	try
    {
        RoadRunnerThread* aThread = (RoadRunnerThread*) handle;
        if(aThread)
        {
			aThread->waitForFinish();
            return true;
        }
		return false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv waitForJobs(RRJobsHandle handle)
{
	try
    {
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
			aTP->waitForFinish();
            return true;
        }
		return false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv isJobFinished(RRJobHandle handle)
{
	try
    {
        RoadRunnerThread* aT = (RoadRunnerThread*) handle;
        if(aT)
        {
			return ! aT->isActive();
        }
		return false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv areJobsFinished(RRJobsHandle handle)
{
	try
    {
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
			return ! aTP->isWorking();
        }
		return false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

int rrCallConv getNumberOfRemainingJobs(RRJobHandle handle)
{
	try
    {
        ThreadPool* aTP = (ThreadPool*) handle;
        if(aTP)
        {
            return aTP->getNumberOfRemainingJobs();
        }
    	return -1;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return -1;
    }
}

bool rrCallConv loadSBML(RRHandle handle, const char* sbml)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
        return rri->loadSBML(sbml, true);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv loadSBMLE(RRHandle handle, const char* sbml, bool forceRecompilation)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
        if(!rri->loadSBML(sbml, forceRecompilation))
        {
            setError("Failed to load SBML semantics");
            return false;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}


bool rrCallConv loadSimulationSettings(RRHandle handle, const char* fileName)
{
	try
    {
        //Check first if file exists first
        if(!FileExists(fileName))
        {
            stringstream msg;
            msg<<"The file "<<fileName<<" was not found";
            setError(msg.str());
            return false;
        }

       	RoadRunner* rri = castFrom(handle);

        if(!rri->loadSimulationSettings(fileName))
        {
            setError("Failed to load simulation settings");
            return false;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

char* rrCallConv getSBML(RRHandle handle)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
        return createText(rri->getSBML());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return NULL;
    }
}

bool rrCallConv unLoadModel(RRHandle handle)
{
	try
    {
      	RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        return rri->unLoadModel();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return NULL;
    }
}

bool rrCallConv setTimeStart(RRHandle handle, const double timeStart)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setTimeStart(timeStart);
    	return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv setTimeEnd(RRHandle handle, const double timeEnd)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setTimeEnd(timeEnd);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv setNumPoints(RRHandle handle, const int nrPoints)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setNumPoints(nrPoints);
	    return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getTimeStart(RRHandle handle, double* timeStart)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return false;
        }

		*timeStart = rri->getTimeStart();
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    }
  	return false;
}

bool rrCallConv getTimeEnd(RRHandle handle, double* timeEnd)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
		*timeEnd = rri->getTimeEnd();
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getNumPoints(RRHandle handle, int* numPoints)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
		*numPoints = rri->getNumPoints();
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setTimeCourseSelectionList(RRHandle handle, const char* list)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
        rri->setTimeCourseSelectionList(list);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv createTimeCourseSelectionList(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        return rri->createTimeCourseSelectionList() > 0 ? true : false;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return false;
    }
}

RRStringArrayHandle rrCallConv getTimeCourseSelectionList(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        StringList sNames = rri->getTimeCourseSelectionList();

        if(!sNames.Count())
        {
            return NULL;
        }

        return createList(sNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }

}

RRResultHandle rrCallConv simulate(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);

        if(!rri->simulate2())
        {
            return NULL;
        }

        SimulationData result = rri->getSimulationResult();

        //Extract the data and return struct..
        RRResult* aResult  = new RRResult;
        aResult->ColumnHeaders = new char*[result.cSize()];
        for(int i = 0; i < result.cSize(); i++)
        {
            aResult->ColumnHeaders[i] = createText(result.getColumnNames()[i]);
        }

        aResult->RSize = result.rSize();
        aResult->CSize = result.cSize();
        int size = aResult->RSize*aResult->CSize;
        aResult->Data = new double[size];

        int index = 0;
        //The data layout is simple row after row, in one single long row...
        for(int row = 0; row < aResult->RSize; row++)
        {
            for(int col = 0; col < aResult->CSize; col++)
            {
                aResult->Data[index++] = result(row, col);
            }
        }
	    return aResult;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRResultHandle rrCallConv getSimulationResult(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);

        SimulationData result = rri->getSimulationResult();

        //Extract the data and return struct..
        RRResult* aResult  = new RRResult;
        aResult->ColumnHeaders = new char*[result.cSize()];
        for(int i = 0; i < result.cSize(); i++)
        {
            aResult->ColumnHeaders[i] = createText(result.getColumnNames()[i]);
            //new char(32);
            //strcpy(aResult->ColumnHeaders[i], result.GetColumnNames()[i].c_str());
        }

        aResult->RSize = result.rSize();
        aResult->CSize = result.cSize();
        int size = aResult->RSize*aResult->CSize;
        aResult->Data = new double[size];

        int index = 0;
        //The data layout is simple row after row, in one single long row...
        for(int row = 0; row < aResult->RSize; row++)
        {
            for(int col = 0; col < aResult->CSize; col++)
            {
                aResult->Data[index++] = result(row, col);
            }
        }
	    return aResult;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRJobHandle rrCallConv simulateJob(RRHandle rrHandle)
{
	try
    {
        RoadRunner *rr = castFrom(rrHandle);
        SimulateThread *t = new SimulateThread();

        if(!t)
        {
            setError("Failed to create a Simulate Thread Pool");
        }
        t->addJob(rr);
        t->start();
        return t;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRJobHandle rrCallConv simulateJobs(RRInstanceListHandle _handles, int nrOfThreads)
{
	try
    {
        RoadRunnerList *rrs = getRRList(_handles);
        Simulate* tp = new Simulate(*rrs, nrOfThreads);

        if(!tp)
        {
            setError("Failed to create a Simulate Thread Pool");
        }
        return tp;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRResultHandle rrCallConv simulateEx(RRHandle handle, const double timeStart, const double timeEnd, const int numberOfPoints)
{
	try
    {
        setTimeStart(handle, timeStart);
        setTimeEnd (handle, timeEnd);
        setNumPoints(handle, numberOfPoints);
	  	return simulate(handle);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRStringArrayHandle rrCallConv getReactionIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        StringList rNames = rri->getReactionIds();

        if(!rNames.Count())
        {
            return NULL;
        }


        return createList(rNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRVectorHandle rrCallConv getRatesOfChange(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        vector<double> rates = rri->getRatesOfChange();

        if(!rates.size())
        {
            return NULL;
		}

        RRVector* list = new RRVector;
        list->Count = rates.size();
        list->Data = new double[list->Count];

        for(int i = 0; i < list->Count; i++)
        {
            list->Data[i] = rates[i];
		}
		return list;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
		setError(msg.str());
		return NULL;
	}
}

RRStringArrayHandle rrCallConv getRatesOfChangeIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        StringList rNames = rri->getRateOfChangeIds();

        if(!rNames.Count())
        {
            return NULL;
        }

        return createList(rNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    }
	return NULL;
}

RRMatrixHandle rrCallConv getUnscaledElasticityMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

		DoubleMatrix tempMat = rri->getUnscaledElasticityMatrix();

        RRMatrixHandle matrix = createMatrix(&tempMat);
	    return matrix;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getScaledElasticityMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


		DoubleMatrix tempMat = rri->getScaledReorderedElasticityMatrix();


        RRMatrixHandle matrix = createMatrix(&tempMat);
	    return matrix;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv getValue(RRHandle handle, const char* symbolId, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
	    *value = rri->getValue(symbolId);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

bool rrCallConv setValue(RRHandle handle, const char* symbolId, const double value)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
    	return rri->setValue(symbolId, value);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}


RRMatrixHandle rrCallConv getStoichiometryMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        DoubleMatrix tempMat = rri->getStoichiometryMatrix();

        RRMatrixHandle matrix = new RRMatrix;
        matrix->RSize = tempMat.RSize();
        matrix->CSize = tempMat.CSize();
        matrix->Data =  new double[tempMat.RSize()*tempMat.CSize()];

        int index = 0;
        for(rr::u_int row = 0; row < tempMat.RSize(); row++)
        {
            for(rr::u_int col = 0; col < tempMat.CSize(); col++)
            {
                matrix->Data[index++] = tempMat(row,col);
            }
        }
	    return matrix;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getConservationMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        DoubleMatrix tempMat = rri->getConservationMatrix();

        RRMatrixHandle matrix = new RRMatrix;
        matrix->RSize = tempMat.RSize();
        matrix->CSize = tempMat.CSize();
        matrix->Data =  new double[tempMat.RSize()*tempMat.CSize()];

        int index = 0;
        for(rr::u_int row = 0; row < tempMat.RSize(); row++)
        {
            for(rr::u_int col = 0; col < tempMat.CSize(); col++)
            {
                matrix->Data[index++] = tempMat(row,col);
            }
        }
	    return matrix;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getLinkMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        DoubleMatrix *tempMat = rri->getLinkMatrix();

		return createMatrix(tempMat);
	}
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getL0Matrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);

        DoubleMatrix *tempMat = rri->getL0Matrix();

		return createMatrix(tempMat);
	}
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getNrMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        DoubleMatrix *tempMat = rri->getNrMatrix();

		return createMatrix(tempMat);
	}
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

C_DECL_SPEC bool rrCallConv hasError()
{
    return (gLastError != NULL) ? true : false;
}

char* rrCallConv getLastError()
{
	if(!gLastError)
    {
	    gLastError = createText("No Error");
    }
    return gLastError;
}

bool rrCallConv reset(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);

        rri->reset();
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

int rrCallConv getNumberOfReactions(RRHandle handle)
{
 	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
           setError(ALLOCATE_API_ERROR_MSG);
           return -1;
        }
        return rri->getNumberOfReactions();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return -1;
    }
}

bool rrCallConv getReactionRate(RRHandle handle, const int rateNr, double* value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getReactionRate(rateNr);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

RRVectorHandle rrCallConv getReactionRates(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);

        vector<double> vec =  rri->getReactionRates();

        RRVector* aVec = createVectorFromVector_double(vec);
        return aVec;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

int rrCallConv getNumberOfBoundarySpecies(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }
        return rri->getNumberOfBoundarySpecies();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return -1;
    }
}

RRStringArrayHandle rrCallConv getBoundarySpeciesIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        StringList bNames = rri->getBoundarySpeciesIds();

        if(!bNames.Count())
        {
            return NULL;
        }

        return createList(bNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	   	return NULL;
    }
}

int rrCallConv getNumberOfFloatingSpecies(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }
        return rri->getNumberOfFloatingSpecies();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	   	return -1;
    }
}

RRStringArrayHandle rrCallConv getFloatingSpeciesIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        StringList fNames = rri->getFloatingSpeciesIds();

        if(!fNames.Count())
        {
            return NULL;
        }

        return createList(fNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

int rrCallConv getNumberOfGlobalParameters(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }
        return rri->getNumberOfGlobalParameters();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	   	return -1;
    }
}

RRStringArrayHandle rrCallConv getGlobalParameterIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        StringList pNames = rri->getGlobalParameterIds();

        if(!pNames.Count())
        {
            return NULL;
        }

        return createList(pNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	   	return NULL;
    }
}

RRVectorHandle rrCallConv getFloatingSpeciesConcentrations(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        vector<double> vec =  rri->getFloatingSpeciesConcentrations();
        RRVector* aVec = createVectorFromVector_double(vec);
        return aVec;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRVectorHandle rrCallConv getBoundarySpeciesConcentrations(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        vector<double> vec =  rri->getBoundarySpeciesConcentrations();
        RRVector* aVec = createVectorFromVector_double(vec);
        return aVec;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}


RRVectorHandle rrCallConv getFloatingSpeciesInitialConcentrations(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        vector<double> vec =  rri->getFloatingSpeciesInitialConcentrations();
        RRVector* aVec = createVectorFromVector_double(vec);
        return aVec;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

bool rrCallConv setFloatingSpeciesByIndex (RRHandle handle, const int index, const double value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setFloatingSpeciesByIndex(index, value);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setBoundarySpeciesByIndex (RRHandle handle, const int index, const double value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setBoundarySpeciesByIndex(index, value);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setGlobalParameterByIndex(RRHandle handle, const int index, const double value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setGlobalParameterByIndex(index, value);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setFloatingSpeciesInitialConcentrations(RRHandle handle, const RRVector* vec)
{
	try
    {
        vector<double> tempVec;
        copyVector(vec, tempVec);
        RoadRunner* rri = castFrom(handle);
        rri->changeInitialConditions(tempVec);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setFloatingSpeciesConcentrations(RRHandle handle, const RRVector* vec)
{
	try
    {
        RoadRunner* rri = castFrom(handle);


        vector<double> tempVec;
        copyVector(vec, tempVec);
        rri->setFloatingSpeciesConcentrations(tempVec);

        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv setBoundarySpeciesConcentrations(RRHandle handle, const RRVector* vec)
{
	try
    {
        vector<double> tempVec;
        copyVector(vec, tempVec);
        RoadRunner* rri = castFrom(handle);
        rri->setBoundarySpeciesConcentrations(tempVec);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv oneStep(RRHandle handle, const double currentTime, const double stepSize, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->oneStep(currentTime, stepSize);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

RRVectorHandle rrCallConv getGlobalParameterValues(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        vector<double> vec =  rri->getGlobalParameterValues();
        RRVector* aVec = createVectorFromVector_double(vec);
        return aVec;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRListHandle rrCallConv getAvailableTimeCourseSymbols(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        NewArrayList slSymbols = rri->getAvailableTimeCourseSymbols();
		return createList(slSymbols);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRListHandle rrCallConv getAvailableSteadyStateSymbols(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        NewArrayList slSymbols = rri->getAvailableSteadyStateSymbols();
		return createList(slSymbols);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

bool rrCallConv getBoundarySpeciesByIndex (RRHandle handle, const int index, double* value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getBoundarySpeciesByIndex(index);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getFloatingSpeciesByIndex (RRHandle handle, const int index, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getFloatingSpeciesByIndex(index);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getGlobalParameterByIndex (RRHandle handle, const int index, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getGlobalParameterByIndex(index);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getuCC (RRHandle handle, const char* variable, const char* parameter, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return false;
		}

        *value = rri->getuCC(variable, parameter);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
	}
}


bool rrCallConv getCC (RRHandle handle, const char* variable, const char* parameter, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getCC(variable, parameter);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv getuEE(RRHandle handle, const char* name, const char* species, double* value)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
        *value = rri->getuEE(name, species);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  		return false;
	}
}

bool rrCallConv getEE(RRHandle handle, const char* name, const char* species, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getEE(name, species);
        return true;
    }
	catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

int rrCallConv getNumberOfDependentSpecies(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }

        return rri->getNumberOfDependentSpecies();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return -1;
    }
}

int rrCallConv getNumberOfIndependentSpecies(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }

        return rri->getNumberOfIndependentSpecies();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return -1;
    }
}


bool rrCallConv steadyState(RRHandle handle, double* value)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);
	   	*value = rri->steadyState();
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	  	return false;
    }
}

bool rrCallConv evalModel(RRHandle handle)
{
	try
	{
        RoadRunner* rri = castFrom(handle);
		if(!rri)
		{
			setError(ALLOCATE_API_ERROR_MSG);
			return false;
		}
		rri->evalModel();
        return true;
	}
	catch(Exception& ex)
	{
		stringstream msg;
		msg<<"RoadRunner exception: "<<ex.what()<<endl;
		setError(msg.str());
	    return false;
	}
}

char* rrCallConv getParamPromotedSBML(RRHandle handle, const char* sArg)
{
	try
	{
        RoadRunner* rri = castFrom(handle);
		if(!rri)
		{
			setError(ALLOCATE_API_ERROR_MSG);
			return NULL;
		}

		string param =  rri->getParamPromotedSBML(sArg);

		char* text = createText(param.c_str());
		return text;
	}
	catch(Exception& ex)
	{
		stringstream msg;
		msg<<"RoadRunner exception: "<<ex.what()<<endl;
		setError(msg.str());
		return NULL;
	}
}

RRVectorHandle rrCallConv computeSteadyStateValues(RRHandle handle)
{
	try
	{
        RoadRunner* rri = castFrom(handle);
		if(!rri)
		{
			setError(ALLOCATE_API_ERROR_MSG);
			return NULL;
		}
		vector<double> vec =  rri->computeSteadyStateValues();

		RRVector* aVec = createVectorFromVector_double(vec);
		return aVec;
	}
	catch(Exception& ex)
	{
		stringstream msg;
		msg<<"RoadRunner exception: "<<ex.what()<<endl;
		setError(msg.str());
		return NULL;
	}
}

bool rrCallConv setSteadyStateSelectionList(RRHandle handle, const char* list)
{
	try
	{
       	RoadRunner* rri = castFrom(handle);
        StringList aList(list, " ,");
        rri->setSteadyStateSelectionList(aList);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return false;
    }
}

RRStringArrayHandle rrCallConv getSteadyStateSelectionList(RRHandle handle)
{
	try
    {
       	RoadRunner* rri = castFrom(handle);
        StringList sNames = rri->getSteadyStateSelectionList();

        if(sNames.Count() == 0)
        {
            return NULL;
        }

        return createList(sNames);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getFullJacobian(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        DoubleMatrix tempMat = rri->getFullJacobian();
        return createMatrix(&tempMat);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getReducedJacobian(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        DoubleMatrix tempMat = rri->getReducedJacobian();
        return createMatrix(&tempMat);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRMatrixHandle rrCallConv getEigenvalues(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
		DoubleMatrix tempMat = rri->getEigenvalues();
        return createMatrix(&tempMat);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

//Todo: this routine should not need a roadrunner handle
RRMatrixHandle rrCallConv getEigenvaluesMatrix (RRHandle handle, const RRMatrixHandle mat)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

		if (mat == NULL) {
         	stringstream msg;
    	    msg<<"RoadRunner exception: "<< "Matrix argument to getEigenvaluesMAtrix is NULL" <<endl;
            setError(msg.str());
			return NULL;
		}

    	// Convert RRMatrixHandle mat to a DoubleMatrix
		DoubleMatrix dmat (mat->RSize, mat->CSize);
		double value;
		for (int i=0; i<mat->RSize; i++)
        {
			for (int j=0; j<mat->CSize; j++)
            {
				getMatrixElement (mat, i, j, &value);
				dmat(i,j) = value;
			}
        }
		DoubleMatrix tempMat = rri->getEigenvaluesFromMatrix (dmat);
        // Convert the DoubleMatrix result to a RRMatrixHandle type
		return createMatrix(&tempMat);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

char* rrCallConv getCSourceFileName(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
    	if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        CGenerator* generator = rri->getCGenerator();
        if(!generator)
        {
            return NULL;
        }

        string fNameS = generator->getSourceCodeFileName();

        fNameS = ExtractFileNameNoExtension(fNameS);
		return createText(fNameS);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

RRCCode* rrCallConv getCCode(RRHandle handle)
{
	try
    {
    	RoadRunner* rri = castFrom(handle);

        CGenerator* generator = rri->getCGenerator();
        if(!generator)
        {
            return NULL;
        }

        RRCCode* cCode = new RRCCode;
		cCode->Header = NULL;
		cCode->Source = NULL;
        string header = generator->getHeaderCode();
        string source = generator->getSourceCode();

        if(header.size())
        {
            cCode->Header = createText(header);
        }

        if(source.size())
        {
            cCode->Source = createText(source);
        }
        return cCode;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

// *******  Not yet implemented  ********
// codeGenerationMode = 0 if mode is C code generation
// codeGenerationMode = 1 ig mode is internal math interpreter
bool rrCallConv setCodeGenerationMode (int _mode)
{
	return false;
}

//NOM forwarded functions
int rrCallConv getNumberOfRules(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }
        if(!rri->getNOM())
        {
            Log(lWarning)<<"NOM is not allocated.";
        	return -1;
        }
        int value = rri->getNOM()->getNumRules();
        return value;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return -1;
    }
}

bool rrCallConv getScaledFloatingSpeciesElasticity(RRHandle handle, const char* reactionId, const char* speciesId, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getScaledFloatingSpeciesElasticity(reactionId, speciesId);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

RRStringArrayHandle rrCallConv getFloatingSpeciesInitialConditionIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        StringList aList = rri->getFloatingSpeciesInitialConditionIds();
		return createList(aList);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRVectorHandle rrCallConv getRatesOfChangeEx(RRHandle handle, const RRVectorHandle vec)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        vector<double> tempList = createVectorFromRRVector(vec);
        tempList = rri->getRatesOfChangeEx(tempList);
        return createVectorFromVector_double (tempList);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRVectorHandle rrCallConv getReactionRatesEx(RRHandle handle, const RRVectorHandle vec)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        vector<double> tempList = createVectorFromRRVector(vec);
        tempList = rri->getReactionRatesEx(tempList);
        return createVectorFromVector_double(tempList);;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRListHandle rrCallConv getElasticityCoefficientIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        NewArrayList aList = rri->getElasticityCoefficientIds();
        RRListHandle bList = createList(aList);
		return bList;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

bool rrCallConv setCapabilities(RRHandle handle, const char* caps)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!caps)
        {
            return false;
        }
        rri->setCapabilities(caps);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

char* rrCallConv getCapabilities(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        return createText(rri->getCapabilities());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
RRStringArrayHandle rrCallConv getEigenvalueIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

		StringList aList = rri->getEigenvalueIds();
		return createList(aList);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRListHandle rrCallConv getFluxControlCoefficientIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        return createList(rri->getFluxControlCoefficientIds());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRMatrixHandle rrCallConv getUnscaledConcentrationControlCoefficientMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
		DoubleMatrix aMat = rri->getUnscaledConcentrationControlCoefficientMatrix();
        //return createMatrix(&(rri->getUnscaledConcentrationControlCoefficientMatrix()));
        return createMatrix(&(aMat));
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRMatrixHandle rrCallConv getScaledConcentrationControlCoefficientMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
		DoubleMatrix aMat = rri->getScaledConcentrationControlCoefficientMatrix();
        return createMatrix(&(aMat));
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRMatrixHandle rrCallConv getUnscaledFluxControlCoefficientMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }

        //return createMatrix(&(rri->getUnscaledFluxControlCoefficientMatrix()));
		DoubleMatrix aMat = rri->getUnscaledFluxControlCoefficientMatrix();
        return createMatrix(&(aMat));
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRMatrixHandle rrCallConv getScaledFluxControlCoefficientMatrix(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        //return createMatrix(&(rri->getScaledFluxControlCoefficientMatrix()));a
		DoubleMatrix aMat = rri->getScaledFluxControlCoefficientMatrix();
        return createMatrix(&(aMat));
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRListHandle rrCallConv getUnscaledFluxControlCoefficientIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
		NewArrayList arrList = rri->getUnscaledFluxControlCoefficientIds();
        return createList(arrList);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

RRList* rrCallConv getConcentrationControlCoefficientIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        NewArrayList list = rri->getConcentrationControlCoefficientIds();
        return createList(list);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

RRListHandle rrCallConv getUnscaledConcentrationControlCoefficientIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        return createList(rri->getUnscaledConcentrationControlCoefficientIds());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

int rrCallConv getNumberOfCompartments(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return -1;
        }
        return rri->getNumberOfCompartments();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return -1;
    }
}

bool rrCallConv getCompartmentByIndex(RRHandle handle, const int index, double *value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return false;
        }
        *value = rri->getCompartmentByIndex(index);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return false;
    }
}

bool rrCallConv setCompartmentByIndex (RRHandle handle, const int index, const double value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        rri->setCompartmentByIndex(index, value);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return false;
    }
}

RRStringArrayHandle rrCallConv getCompartmentIds(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        if(!rri)
        {
            setError(ALLOCATE_API_ERROR_MSG);
            return NULL;
        }
        return createList(rri->getCompartmentIds());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

bool rrCallConv getRateOfChange(RRHandle handle, const int index, double* value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        *value = rri->getRateOfChange(index);
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

// Utility functions ==========================================================
int rrCallConv getNumberOfStringElements (const RRStringArrayHandle list)
{
	if (!list)
		return (-1);
	else
	    return (list->Count);
}

char* rrCallConv getStringElement (RRStringArrayHandle list, int index)
{
	try 
	{
	  if (list == NULL)
	  {
	     return NULL;
	  }

	  if ((index < 0) || (index >= list->Count)) 
	  {
         setError("Index out of range");
         return NULL;
	  }
    
	  return createText(list->String[index]);
	}
	catch(Exception& ex)
	{
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

char* rrCallConv stringArrayToString (const RRStringArrayHandle list)
{
	try
    {
        if(!list)
        {
            return NULL;
        }

		stringstream resStr;
	    for(int i = 0; i < list->Count; i++)
        {
        	resStr<<list->String[i];;
            if(i < list->Count -1)
            {
            	resStr <<" ";
            }
        }

    	return createText(resStr.str());
    }
    catch(Exception& ex)
    {
        stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
		return NULL;
    }
}

char* rrCallConv resultToString(const RRResultHandle result)
{
	try
    {
        if(!result)
        {
            return NULL;
        }
		stringstream resStr;
		//RRResult is a 2D matrix, and column headers (strings)
        //First header....
	    for(int i = 0; i < result->CSize; i++)
        {
        	resStr<<result->ColumnHeaders[i];
            if(i < result->CSize -1)
            {
            	resStr <<"\t";
            }
        }
        resStr<<endl;

        //Then the data
        int index = 0;
	    for(int j = 0; j < result->RSize; j++)
   	    {
		    for(int i = 0; i < result->CSize; i++)
    	    {
        		resStr<<result->Data[index++];
	            if(i < result->CSize -1)
    	        {
        	    	resStr <<"\t";
            	}
            }
	    	resStr <<"\n";
        }
        return createText(resStr.str());
    }
    catch(Exception& ex)
    {
        stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return NULL;
    }
}

char* rrCallConv matrixToString(const RRMatrixHandle matrixHandle)
{
	try
    {
        if(!matrixHandle)
        {
            return NULL;
        }

        RRMatrix& mat = *matrixHandle;
        stringstream ss;
        //ss<<"\nmatrix dimension: "<<mat.RSize<<"x"<<mat.CSize<<" --\n";
        ss<<"\n";
        for(int row = 0; row < mat.RSize; row++)
        {
            for(int col = 0; col < mat.CSize; col++)
            {
                ss<<mat.Data[row*mat.CSize + col];
                if(col < mat.CSize + 1)
                {
                    ss<<"\t";
                }
            }
            ss<<endl;
        }
        return createText(ss.str());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

char* rrCallConv vectorToString(RRVectorHandle vecHandle)
{
	try
    {
        if(!vecHandle)
        {
            setError("Null vector in vectorToString");
            return NULL;
        }

        RRVector& vec = *vecHandle;

        stringstream ss;
        ss<<"vector dimension: "<<vec.Count<<" \n";

        for(int index = 0; index < vec.Count; index++)
        {
            ss<<vec.Data[index];
            if(index < vec.Count + 1)
            {
                ss<<"\t";
            }
        }
        ss<<endl;
        return createText(ss.str());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return NULL;
    }
}

// Free Functions =====================================================
bool rrCallConv freeRRInstance(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
    	delete rri;
        rri = NULL;
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv freeMatrix(RRMatrixHandle matrix)
{
	try
    {
        if(matrix)
        {
            delete [] (matrix->Data);
            delete matrix;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv freeResult(RRResultHandle handle)
{
	try
    {
        delete handle;
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv freeText(char* text)
{
	try
    {
        delete [] text;
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    	return false;
    }
}

bool rrCallConv freeStringArray(RRStringArrayHandle sl)
{
	try
    {
    	delete sl;
    	return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv freeVector(RRVectorHandle vector)
{
	try
    {
        if(vector)
        {
    	   delete [] vector->Data;
        }
        return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv freeCCode(RRCCodeHandle code)
{
	try
    {
        if(code)
        {
            delete code->Header;
            delete code->Source;
        }
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

/////////////////////////////////////////////////////////////
void rrCallConv pause()
{
    rr::Pause(true, "Hit any key to continue..\n");
}

RRVectorHandle rrCallConv createVector (int size)
{
   RRVectorHandle list = new RRVector;
   list->Count = size;
   list->Data = new double[list->Count];
   return list;
}

int rrCallConv getVectorLength (RRVectorHandle vector)
{
	if (vector == NULL)
    {
		setError ("Vector argument is null in getVectorLength");
		return -1;
	}
	else
    {
		return vector->Count;
    }
}

bool rrCallConv getVectorElement (RRVectorHandle vector, int index, double *value)
{
	if (vector == NULL)
    {
		setError ("Vector argument is null in getVectorElement");
		return false;
	}

	if ((index < 0) || (index >= vector->Count))
    {
		stringstream msg;
		msg << "Index out range in getVectorElement: " << index;
        setError(msg.str());
		return false;
	}

	*value = vector->Data[index];
	return true;
}

bool rrCallConv setVectorElement (RRVectorHandle vector, int index, double value)
{
	if (vector == NULL)
    {
		setError ("Vector argument is null in setVectorElement");
		return false;
	}

	if ((index < 0) || (index >= vector->Count))
    {
		stringstream msg;
		msg << "Index out range in setVectorElement: " << index;
        setError(msg.str());
		return false;
	}

	vector->Data[index] = value;
	return true;
}

// Matrix Routines
// ------------------------------------------------------------------------------------

RRMatrixHandle rrCallConv createRRMatrix (int r, int c)
{
   	RRMatrixHandle matrix = new RRMatrix;
   	matrix->RSize = r;
   	matrix->CSize = c;
   	int dim =  matrix->RSize * matrix->CSize;
   	if(dim)
   	{
		matrix->Data =  new double[dim];
		return matrix;
   	}
   	else
	{
        delete matrix;
		setError ("Dimensions for new RRMatrix in createRRMatrix are zero");
        return NULL;
   }
}

int rrCallConv getMatrixNumRows (RRMatrixHandle m)
{
	if (m == NULL) {
		setError ("Matrix argument is null in getMatrixNumRows");
		return -1;
	}
	return m->RSize;
}

int  rrCallConv getMatrixNumCols (RRMatrixHandle m)
{
	if (m == NULL) {
		setError ("Matrix argument is null in getMatrixNumCols");
		return -1;
	}

	return m->CSize;
}

bool rrCallConv getMatrixElement (RRMatrixHandle m, int r, int c, double* value)
{
	if (m == NULL)
    {
		setError ("Matrix argument is null in getMatrixElement");
		return false;
	}

	if ((r < 0) || (c < 0) || (r >= m->RSize) || (c >= m->CSize))
    {
		stringstream msg;
		msg << "Index out range in getMatrixElement: " << r << ", " << c;
        setError(msg.str());
		return false;
	}

	*value = m->Data[r*m->CSize + c];
	return true;
}

bool rrCallConv setMatrixElement (RRMatrixHandle m, int r, int c, double value)
{
	if (m == NULL)
    {
		setError ("Matrix argument is null in setMatrixElement");
	    return false;
	}

	if ((r < 0) || (c < 0) || (r >= m->RSize) || (c >= m->CSize))
    {
		stringstream msg;
		msg << "Index out range in setMatrixElement: " << r << ", " << c;
        setError(msg.str());
		return false;
	}

	m->Data[r*m->CSize + c] = value;
	return true;
}

int rrCallConv  getResultNumRows (RRResultHandle result)
{
	if (result == NULL)
    {
       setError ("result argument is null in getResultNumRows");
       return -1;
	}
	return result->RSize;
}

int  rrCallConv  getResultNumCols (RRResultHandle result)
{
	if (result == NULL)
    {
       setError ("result argument is null in getResultNumCols");
       return -1;
	}
	return result->CSize;
}

bool  rrCallConv getResultElement(RRResultHandle result, int r, int c, double *value)
{
	if (result == NULL)
    {
	   setError ("result argument is null in getResultElement");
       return false;
	}

	if ((r < 0) || (c < 0) || (r >= result->RSize) || (c >= result->CSize))
    {
		stringstream msg;
		msg << "Index out range in getResultElement: " << r << ", " << c;
        setError(msg.str());
		return false;
    }

	*value = result->Data[r*result->CSize + c];
	return true;
}

char*  rrCallConv getResultColumnLabel (RRResultHandle result, int column)
{
	if (result == NULL)
    {
	   setError ("result argument is null in getResultColumnLabel");
       return NULL;
	}

	if ((column < 0) || (column >= result->CSize))
    {
		stringstream msg;
		msg << "Index out range in getResultColumnLabel: " << column;
        setError(msg.str());
		return NULL;
    }

	return result->ColumnHeaders[column];
}

char* rrCallConv getCCodeHeader(RRCCodeHandle code)
{
	if (code == NULL)
    {
    	setError ("code argument is null in getCCodeHeader");
		return NULL;
    }
	return code->Header;
}

char* rrCallConv getCCodeSource(RRCCodeHandle code)
{
	if (code == NULL)
    {
        setError ("code argument is null in getCCodeSource");
		return NULL;
    }
	return code->Source;
}

// -------------------------------------------------------------------
// List Routines
// -------------------------------------------------------------------
RRListHandle rrCallConv createRRList()
{
	RRListHandle list = new RRList;
	list->Count = 0;
	list->Items = NULL;
	return list;
}

void rrCallConv freeRRList (RRListHandle theList)
{
	if(!theList)
    {
    	return;
    }
    int itemCount = theList->Count;
    for(int i = 0; i < itemCount; i++)
    {
        if(theList->Items[i]->ItemType == litString)
        {
              delete [] theList->Items[i]->data.sValue;
        }
        if(theList->Items[i]->ItemType == litList)
        {
            freeRRList ((RRList *) theList->Items[i]->data.lValue);
        }
        delete theList->Items[i];
    }
	delete [] theList->Items;
    delete theList;
    theList = NULL;
}

RRListItemHandle rrCallConv createIntegerItem (int value)
{
	RRListItemHandle item =  new RRListItem;
	item->ItemType = litInteger;
	item->data.iValue = value;
	return item;
}

RRListItemHandle rrCallConv createDoubleItem (double value)
{
	RRListItemHandle item = new RRListItem;
	item->ItemType = litDouble;
	item->data.dValue = value;
	return item;
}

RRListItemHandle rrCallConv createStringItem (char* value)
{
	RRListItemHandle item = new RRListItem;
	item->ItemType = litString;
	item->data.sValue = createText(value);
	return item;
}

RRListItemHandle rrCallConv createListItem (RRList* value)
{
	RRListItemHandle item = new RRListItem;
	item->ItemType = litList;
	item->data.lValue = value;
	return item;
}

// Add an item to a given list, returns the index to
// the item in the list. Returns -1 if it fails.
int rrCallConv addItem (RRListHandle list, RRListItemHandle *item)
{
	int n = list->Count;

	RRListItemHandle *newItems = new RRListItemHandle [n+1];
    if(!newItems)
    {
    	setError("Failed allocating memory in addItem()");
    	return -1;
    }

    for(int i = 0; i < n; i++)
    {
    	newItems[i] = list->Items[i];
    }

    newItems[n] = *item;
    RRListItemHandle *oldItems = list->Items;
    list->Items = newItems;

    delete [] oldItems;

	list->Count = n+1;
	return n;
}

bool rrCallConv isListItemInteger (RRListItemHandle item)
{
	return (item->ItemType == litInteger) ? true : false;
}

bool rrCallConv isListItemDouble (RRListItemHandle item)
{
	return (item->ItemType == litDouble) ? true : false;
}

bool rrCallConv isListItemString (RRListItemHandle item)
{
	return (item->ItemType == litString) ? true : false;
}

bool rrCallConv isListItemList (RRListItemHandle item)
{
	return (item->ItemType == litList) ? true : false;
}

RRListItemHandle rrCallConv getListItem (RRListHandle list, int index)
{
	return (index >= list->Count) ? NULL : list->Items[index];
}

bool rrCallConv getIntegerListItem (RRListItemHandle item, int *value)
{
    if (item->ItemType == litInteger)
    {
        *value = item->data.iValue;
        return true;
    }
    return false;
}

bool rrCallConv getDoubleListItem (RRListItemHandle item, double *value)
{
    if (item->ItemType == litDouble)
    {
    	*value = item->data.dValue;
     	return true;
    }

    return false;
}

char* rrCallConv getStringListItem (RRListItemHandle item)
{
	return (item->ItemType == litString) ? item->data.sValue : NULL;
}

RRListHandle rrCallConv getList (RRListItemHandle item)
{
	return (item->ItemType == litList) ? item->data.lValue : NULL;
}

bool rrCallConv isListItem (RRListItemHandle item, ListItemType itemType)
{
	return  (item->ItemType == itemType) ? true : false;
}

int rrCallConv getListLength (RRListHandle myList)
{
	return myList->Count;
}

char* rrCallConv listToString (RRListHandle list)
{
	try
    {
        if(!list)
        {
            return NULL;
        }

        //Types of list items
        char*           cVal;
        int             intVal;
        double          dVal;
        RRList*        lVal; 		//list is nested list
		stringstream resStr;
        resStr<<"{";
	    for(int i = 0; i < list->Count; i++)
        {
			switch(list->Items[i]->ItemType)
            {
                case litString:
					cVal = list->Items[i]->data.sValue;
                    resStr<<"\"";
                    if(cVal)
                    {
                    	resStr<<cVal;
                    }
                    resStr<<"\"";
                break;

                case litInteger:
					intVal = list->Items[i]->data.iValue;
                    resStr<< (intVal);
                break;

                case litDouble:
					dVal =  list->Items[i]->data.dValue;
                    resStr<< (dVal);
                break;

                case litList:
					lVal = list->Items[i]->data.lValue;
                    if(lVal)
                    {
                    	char* text = listToString(lVal);
                    	resStr<<text;
                        freeText(text);
                    }
                    else
                    {
						resStr<<"{}";
                    }
                break;
            }

            if(i < list->Count -1)
            {
                resStr<<",";
            }
        }
        resStr<<"}";
        return createText(resStr.str());

    }
    catch(Exception& ex)
    {
        stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
        return NULL;
    }
}
//====================== DATA WRITING ROUTINES ======================
bool rrCallConv writeRRData(RRHandle rrHandle, const char* fileNameAndPath)
{
	try
    {
        RoadRunner *rr = castFrom(rrHandle);
        SimulationData data;
        data = rr->getSimulationResult();

        data.writeTo(fileNameAndPath);
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}

bool rrCallConv writeMultipleRRData(RRInstanceListHandle rrHandles, const char* fileNameAndPath)
{
	try
    {
        RoadRunnerList *rrs = getRRList(rrHandles);

        int rrCount = rrs->count();
        SimulationData allData;
        for(int i = rrCount -1 ; i >-1 ; i--) //"Backwards" because bad plotting program..
        {
            RoadRunner* rr = (*rrs)[i];
            if(rr)
            {
            	SimulationData data = rr->getSimulationResult();
	            allData.append(data);
            }
        }

        allData.writeTo(fileNameAndPath);
		return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
	    return false;
    }
}


//PLUGIN Functions
bool rrCallConv loadPlugins(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
    	return rri->getPluginManager().load();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return false;
    }
}

bool rrCallConv unLoadPlugins(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
    	return rri->getPluginManager().unload();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return false;
    }
}

int rrCallConv getNumberOfPlugins(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
    	return rri->getPluginManager().getNumberOfPlugins();
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return -1;
    }
}

RRStringArray* rrCallConv getPluginNames(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        StringList names = rri->getPluginManager().getPluginNames();
        return createList(names);
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

RRStringArray* rrCallConv getPluginCapabilities(RRHandle handle, const char* pluginName)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(pluginName);
        if(aPlugin)
        {
        	StringList aList;
            vector<Capability>* caps = aPlugin->getCapabilities();
            if(!caps)
            {
            	return NULL;
            }

            for(int i = 0; i < caps->size(); i++)
            {
            	aList.Add((*caps)[i].getName());
            }
        	return createList(aList);
        }
        else
        {
	        return NULL;
        }
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

RRStringArray* rrCallConv getPluginParameters(RRHandle handle, const char* pluginName, const char* capability)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(pluginName);
        if(aPlugin)
        {
        	StringList aList;
            Parameters* paras = aPlugin->getParameters(capability);
            if(!paras)
            {
            	return NULL;
            }

            for(int i = 0; i < paras->size(); i++)
            {
            	aList.Add((*paras)[i]->getName());
            }
        	return createList(aList);
        }
        else
        {
	        return NULL;
        }
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

RRParameter* rrCallConv getPluginParameter(RRHandle handle, const char* pluginName, const char* parameterName)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(pluginName);
        if(aPlugin)
        {

            rr::BaseParameter *para = aPlugin->getParameter(parameterName);
        	return createParameter( *(para) );
        }
        return NULL;

    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

bool rrCallConv setPluginParameter(RRHandle handle, const char* pluginName, const char* parameterName, const char* value)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(pluginName);
        if(aPlugin)
        {
            return aPlugin->setParameter(parameterName, value);
        }
		return false;

    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

char* rrCallConv getPluginInfo(RRHandle handle, const char* name)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(name);
        if(aPlugin)
        {
        	return createText(aPlugin->getInfo());
        }
        else
        {
	        return createText("No such plugin: " + string(name));
        }
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

bool rrCallConv executePlugin(RRHandle handle, const char* name)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        Plugin* aPlugin = rri->getPluginManager().getPlugin(name);
        if(aPlugin)
        {
        	return aPlugin->execute();
        }
        else
        {
			return false;
        }
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

// LOGGING ROUTINES
void rrCallConv logMsg(CLogLevel lvl, const char* msg)
{
	try
    {
		Log((LogLevel) lvl)<<msg;
    }
    catch(const Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
    }
}

bool rrCallConv enableLoggingToConsole()
{
	try
    {
	    LogOutput::mLogToConsole = true;
    	return true;
    }
    catch(const Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return false;
    }
}

bool rrCallConv enableLoggingToFile(RRHandle handle)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        char* tempFolder = getTempFolder(handle);
		string logFile = JoinPath(tempFolder, "RoadRunner.log") ;
        freeText(tempFolder);

//        gLog.Init("", gLog.GetLogLevel(), unique_ptr<LogFile>(new LogFile(logFile.c_str())));
        gLog.Init("", gLog.GetLogLevel(), new LogFile(logFile.c_str()));
    	return true;
    }
    catch(const Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return false;
    }
}

char* rrCallConv testString (char* testStr)
{
	return testStr;
}

bool rrCallConv setLogLevel(char* _lvl)
{
	try
    {
        LogLevel lvl = GetLogLevel(_lvl);
		gLog.SetCutOffLogLevel(lvl);
    	return true;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return false;
    }
}

char* rrCallConv getLogLevel()
{
	try
    {
        string level = gLog.GetCurrentLogLevel();
        char* lvl = createText(level.c_str());
    	return lvl;
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

char* rrCallConv getLogFileName()
{
	try
    {
    	return createText(gLog.GetLogFileName().c_str());
    }
    catch(Exception& ex)
    {
    	stringstream msg;
    	msg<<"RoadRunner exception: "<<ex.what()<<endl;
        setError(msg.str());
  	    return NULL;
    }
}

char* rrCallConv getBuildDate()
{
    return createText(__DATE__);
}

char* rrCallConv getBuildTime()
{
    return createText(__TIME__);
}

char* rrCallConv getBuildDateTime()
{
    return createText(string(__DATE__) + string(" ") + string(__TIME__));
}


int rrCallConv getInstanceCount(RRInstanceListHandle iList)
{
	return iList->Count;
}

RRHandle rrCallConv getRRHandle(RRInstanceListHandle iList, int index)
{
	return iList->Handle[index];
}


}
//We only need to give the linker the folder where libs are
//using the pragma comment. Works for MSVC and codegear
#if defined(CG_IDE)

#if defined(STATIC_RR)
	#pragma comment(lib, "roadrunner-static.lib")
#else
	#pragma comment(lib, "roadrunner.lib")
#endif

#pragma comment(lib, "rr-libstruct-static.lib")
#pragma comment(lib, "pugi-static.lib")
#pragma comment(lib, "libsbml-static.lib")
#pragma comment(lib, "sundials_cvode.lib")
#pragma comment(lib, "sundials_nvecserial.lib")
#pragma comment(lib, "libxml2_xe.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "poco_foundation-static.lib")
#pragma comment(lib, "nleq-static.lib")
#endif

