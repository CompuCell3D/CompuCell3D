#include <string>
#include "rrLogger.h"
#include "UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
using namespace std;
using namespace UnitTest;

//Add using clauses..
using rr::JoinPath;
using rr::FileExists;

extern RRHandle gRR;	//Global roadrunner C handle
extern string 	gBinPath;
extern string 	gSBMLModelsPath;
extern string 	gCompiler;
extern string 	gSupportCodeFolder;
extern string   gTempFolder;
extern string   gDataOutputFolder;
extern bool 	gDebug;
SUITE(Base)
{
    TEST(AllocateRR)
    {
        if(!gRR)
        {
            gRR = getRRInstance();
        }

        CHECK(gRR!=NULL);	//If gRR == NULL this is a fail

		if(gRR)
		{
            setTempFolder(gTempFolder.c_str());
		}
    }

    TEST(LOGGING)
    {
    	if(gDebug == true)
        {
			CHECK(enableLogging());
        	char* logFName = getLogFileName();
        	CHECK_EQUAL("RoadRunner.log", logFName);
            rr::gLog.Init("", rr::lDebug, unique_ptr<rr::LogFile>(new rr::LogFile(logFName)));
        }
        else
        {
        	rr::gLog.Init("", rr::lDebug, unique_ptr<rr::LogFile>(new rr::LogFile("")));
        	char* logFName = getLogFileName();
        	CHECK_EQUAL("<none>", logFName);
        }
    }

    TEST(VERSIONS)
    {
    	CHECK_EQUAL(getVersion(), 			"1.0.0");
		CHECK_EQUAL(getlibSBMLVersion(), 	"5.6.0");
    }

    TEST(MODEL_FILES)	//Test that model files for the tests are present
    {
    	CHECK(FileExists(JoinPath(gSBMLModelsPath, "feedback.xml")));
    	CHECK(FileExists(JoinPath(gSBMLModelsPath, "ss_threeSpecies.xml")));
        CHECK(FileExists(JoinPath(gSBMLModelsPath, "ss_TurnOnConservationAnalysis.xml")));
        CHECK(FileExists(JoinPath(gSBMLModelsPath, "squareWaveModel.xml")));
    }

    TEST(LOAD_SBML)
    {
		CHECK(gRR!=NULL);
		if(!gRR)
		{
			return;
		}

		string model = JoinPath(gSBMLModelsPath, "ss_threeSpecies.xml");
		CHECK(loadSBMLFromFile(model.c_str()));
	}

	TEST(GET_CAPABILITIES)
	{
		CHECK(gRR!=NULL);
		if(!gRR)
		{
			return;
		}

		string caps(getCapabilities());

	}

	TEST(LISTS)
	{
		RRListHandle myList = createRRList();
		CHECK(myList!=NULL);

        // First construct [5, 3.1415]
        RRListItemHandle myItem = createIntegerItem (5);
        addItem (myList, &myItem);
        myItem = createDoubleItem (3.1415);
        addItem (myList, &myItem);

        // Next construct [5, 3.1415, [2.7182, "Hello"]]
        myItem = createListItem (createRRList());
        addItem (myList, &myItem);
        RRListItemHandle newItem = createDoubleItem (2.7182);
        addItem (getList (myItem), &newItem);
        newItem = createStringItem ("Hello");
        addItem (getList (myItem), &newItem);

        int length = getListLength (myList);
        myItem = getListItem (myList, 0);

       	CHECK(myItem->data.iValue == 5);

        myItem = getListItem (myList, 1);
		CHECK(myItem->data.dValue == 3.1415);
        myItem = getListItem (myList, 2);

        //Item with index 1 is a litDouble!
        CHECK(isListItem(getListItem (myList, 1), litInteger) == false);
        freeRRList (myList);

        //We could check more about lists, but it seem pretty solid at this time..?
    }


//   	TEST(AllocateDeAllocateRR)
// 	{
//     	int memoryBefore = 0;
//         int memoryAfter  = 10;
//         for(int i = 0; i < 1000; i++)
//         {
//             if(gRR)
//             {
//                 freeRRInstance(gRR);
//             }
//
//            	gRR = getRRInstance();
//         }
//
// 		//To check this properly, we will need to measure memory before and after somehow..
// 		CHECK_CLOSE(memoryBefore, memoryAfter, 10);
// 	}

}

