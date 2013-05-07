#pragma hdrstop
#include <stdio.h>
#include "../../rrc_api.h"

/*--------------------------------------------------------------------------
Example showing how to obtain model generated C code

---------------------------------------------------------------------------*/

int main()
{
    RRHandle rrHandle;
	RRCCodeHandle code;
	char* text;
	char* sbml;

    char modelFileName[2048];

	//-------------------------------
    rrHandle =  createRRInstance();
    if(!rrHandle)
    {
        printf("No handle...");
    }

	text = getBuildDate();

	if(text)
	{
		printf("Build date: %s \n", text);
		freeText(text);
	}

    //Setup tempfolder
    strcpy(text, "../temp");
    if(!setTempFolder(rrHandle, text))
    {
    	printf("The temp file folder \'%s\' do not exist. Exiting...\n", text);
        exit(0);
    }

	//Setup logging
   	setLogLevel("Info");
	enableLoggingToConsole();
   	enableLoggingToFile(rrHandle);


	strcpy(modelFileName, "../models/test_1.xml");

	sbml = getFileContent(modelFileName);

    //To get the C Code, the code needs to be generated
    if(!loadSBML(rrHandle, sbml))
    {
    	printf("Failed loading SBML.\n");
        printf("Last error: %s", getLastError());
        printf("Exiting...");
        return -1;
    }

	code = getCCode(rrHandle);
    if(!code)
    {
	  	printf("Failed to get C-code from RoadRunner");
        printf("Exiting...");
        return -1;
    }

    printf("START OF CODE ==========\n");
	if(code->Header)
	{
		printf("C Header =========== \n%s \n\n", code->Header);
	}
	else
	{
		printf("C Header =========== \n is empty!\n");
	}

	if(code->Source)
	{
		printf("C Source =========== \n%s \n", code->Source);
	}
	else
	{
		printf("C Source  =========== \n is empty!\n");
	}

    printf("END OF CODE ==========\n");

	///// Cleanup
    freeCCode(code);
    text = getCopyright();
    if(hasError())
    {
        char* error = getLastError();
        printf("Last error: %s \n", error);
        freeText(error);
    }
    printf(text);
    freeText(text);
    freeRRInstance(rrHandle);
    return 0;
}

#pragma link "rrc_api.lib"

