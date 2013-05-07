#pragma hdrstop
#if defined(linux)
#include <string.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "rrc_api.h"

int main()
{
    //Some Declarations (has to be here because this is C)
	RRHandle 				rrHandle;
    RRJobHandle				jobHandle;
    char tempFolder[1024];
    double val;
	char* modelFileName = "../models/test_1.xml";
   	char  buf[2048];
    char dataFile[1024];
    char msg[1024];
	// -------------------------------------------------------------

	printf("Starting C program...\n");

    rrHandle = createRRInstance();

    if(!rrHandle)
    {
        printf("No handles...\n");
    }
    else
    {
	    printf("Handles allocated succesfully..\n");
    }

   	setLogLevel("Info");
    strcpy(tempFolder, "../temp");
    if(!setTempFolder(rrHandle, tempFolder))
    {
    	printf("The temp file folder \'%s\' do not exist. Exiting...\n", tempFolder);
        exit(0);
    }
	enableLoggingToConsole();
   	enableLoggingToFile(rrHandle);

	//loadSBML models in threads instead
    jobHandle = loadSBMLFromFileJob(rrHandle, modelFileName);

    //waitForJob will block until the thread haa finished
	//Instead, one can could check for activeJob, i.e. non blocking (see below)
   waitForJob(jobHandle);

    //Set parameters
    logMsg(clInfo, " ---------- SETTING PARAMETERS -------------");

    //Setup instances with different variables
    val = 0;
    getValue(rrHandle, "k1", &val);
    setValue(rrHandle, "k1", val/(2.5));
    setNumPoints(rrHandle, 500);
    setTimeEnd(rrHandle, 150);
    setTimeCourseSelectionList(rrHandle, "TIME S1");


    //Simulate
    logMsg(clInfo, " ---------- SIMULATING ---------------------");

    //Simulate them using a pool of threads..
    jobHandle = simulateJob(rrHandle);

    waitForJob(jobHandle);

 	//Write data to a file
    strcpy(dataFile, "oneJobData.dat");
    strcat(msg,"Writing data to file: ");
    strcat(msg, dataFile);
    logMsg(clInfo, msg);
    writeRRData(rrHandle, dataFile);

	// Cleanup
    freeRRInstance(rrHandle);

	if(hasError())
    {
        char* error = getLastError();
        sprintf(buf, "Last error %s \n", error);
    }
	return 0;
}

#if defined(CG_IDE)
#pragma link "rrc_api.lib"
#endif

//Non blocking code waiting for threadpool to finish
//    while(true)
//    {
//		int nrOfRemainingJobs = getNumberOfRemainingJobs(tpHandle);
//        if (nrOfRemainingJobs == 0)
//        {
//           	logMsg(lInfo, "All jobs are done!!!\n");
//        	break;
//        }
//        else
//        {
//        	sprintf(buf, "There are %d remaining jobs\n", nrOfRemainingJobs);
//        	logMsg(lInfo, buf);
//            sleep(0.1);
//        }
//	}

