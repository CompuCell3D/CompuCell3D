#pragma hdrstop
#if defined(linux)
#include <stdlib.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "rrc_api.h"

int main(int argc, char* argv[])
{
   //Some Declarations (has to be here because this is C)
	RRInstanceListHandle 	rrs;
    RRJobsHandle 		    jobsHandle;		//Jobs handle.. use to check when a pool of threads has finished..
    char tempFolder[1024];

	//This path should work on both windows and linux..
	char* modelFileName = "../models/test_1.xml";
    int   handleCount = 10;
    int   threadCount = 1;
    int   i;
   	char  errorBuf[2048];

	printf("Starting C program...\n");
    if(argc > 1)
    {
        handleCount = atoi(argv[1]);
        if(argc > 2)
        {
            threadCount = atoi(argv[2]);
        }
    }

    printf("Allocating %d handles and %d threads\n\n", handleCount, threadCount);
    rrs = createRRInstances(handleCount);

    if(!rrs)
    {
        printf("No handles...\n");
    }
    else
    {
	    printf("%d handles allocated succesfully..\n", handleCount);
    }

   	setLogLevel("Info");
	enableLoggingToConsole();

    strcpy(tempFolder, "../temp");
    for(i = 0; i < handleCount; i++)
    {
        if(!setTempFolder(rrs->Handle[i], tempFolder))
        {
            printf("The temp file folder \'%s\' do not exist. Exiting...\n", tempFolder);
            exit(0);
        }
    }

   	enableLoggingToFile(rrs->Handle[0]);

	//loadSBML models in threads instead
    jobsHandle = loadSBMLFromFileJobs(rrs, modelFileName, threadCount);

    //waitForJobs will block until all threads have finished
	//Instead, one can could check for activeJobs, i.e. non blocking (see below)
    //waitForJobs(jobsHandle);
    //Non blocking code waiting for threadpool to finish
    printf("Entering wait loop...\n");

    //Non blocking code waiting for threadpool to finish
    while(true)
    {
        if (areJobsFinished(jobsHandle) == true)
        {
           	logMsg(clInfo, "All jobs are done!!!\n");
        	break;
        }
        else
        {
        }
	}

    //Set parameters
    logMsg(clInfo, " ---------- SETTING PARAMETERS -------------");

    //Setup instances with different variables
    for(i = 0; i < handleCount; i++)
    {
        double val = 0;
        getValue(rrs->Handle[i], "k1", &val);
        setValue(rrs->Handle[i], "k1", val/(2.5*(i + 1)));
        setNumPoints(rrs->Handle[i], 10);
        setTimeEnd(rrs->Handle[i], 150);
        setTimeCourseSelectionList(rrs->Handle[i], "S1");
    }

    //Simulate
    logMsg(clInfo, " ---------- SIMULATING ---------------------");

    //Simulate them using a pool of threads..

    jobsHandle = simulateJobs(rrs, threadCount);
    printf("Entering wait loop...\n");

    waitForJobs(jobsHandle);

  	//Write data to a file
	writeMultipleRRData(rrs, "allData.dat");

	// Cleanup
    freeRRInstances(rrs);

	if(hasError())
    {
        char* error = getLastError();
        sprintf(errorBuf, "Last error %s \n", error);
    }
	return 0;
}

#if defined(CG_IDE)
#pragma link "rrc_api.lib"
#endif

//Non blocking code waiting for threadpool to finish
//    while(true)
//    {
//		int nrOfRemainingJobs = getNumberOfRemainingJobs(jobsHandle);
//        if (nrOfRemainingJobs == 0)
//        {
//           	logMsg(lInfo, "All jobs are done!!!\n");
//        	break;
//        }
//        else
//        {
//        	sprintf(errorBuf, "There are %d remaining jobs\n", nrOfRemainingJobs);
//        	logMsg(lInfo, errorBuf);
//            sleep(0.1);
//        }
//	}

