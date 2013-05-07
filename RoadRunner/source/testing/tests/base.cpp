#include "unit_test/UnitTest++.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
using namespace UnitTest;
using namespace rr;

extern RoadRunner* 		gRR;
extern string 			gSBMLModelsPath;
extern string 			gCompiler;
extern string 			gSupportCodeFolder;
extern string 			gRRInstallFolder;
extern string 			gTempFolder;
extern vector<string> 	gModels;
SUITE(Base)
{
    TEST(VERSIONS)
    {
    	//Static functions, don't need a handle, (gRR) ?
    	CHECK_EQUAL(gRR->getVersion(), 				"1.0.0");
		CHECK_EQUAL(gRR->getlibSBMLVersion(), 	"5.6.0");
    }

	TEST(AllocateRR)
	{


		if(!gRR)
		{
			gRR = new RoadRunner(gSupportCodeFolder, gCompiler, gTempFolder);
		}

		CHECK(gRR!=NULL);
	}

	TEST(AllocateDeAllocateRR)
	{
    	int memoryBefore = 0;
        int memoryAfter  = 10;
        for(int i = 0; i < 1000; i++)
        {
            if(gRR)
            {
                delete gRR;
                gRR = NULL;
            }
			gRR = new RoadRunner(gSupportCodeFolder, gCompiler, gTempFolder);
        }

		//To check this properly, we will need to measure memory before and after somehow..
		CHECK_CLOSE(memoryBefore, memoryAfter, 10);
	}

    TEST(MODEL_FILES)	//Test that model files for the tests are present
    {
    	//Populate models
        gModels.clear();
        gModels.push_back("feedback.xml");
        gModels.push_back("ss_threeSpecies.xml");
        gModels.push_back("ss_TurnOnConservationAnalysis.xml");
        gModels.push_back("squareWaveModel.xml");
        gModels.push_back("test_1.xml");

    	for(int i = 0 ; i < gModels.size(); i++)
        {
    		CHECK(FileExists(JoinPath(gSBMLModelsPath, gModels[i])));
        }
    }

	TEST(LOAD_SBML)
	{
    	for(int i = 0 ; i < gModels.size(); i++)
        {
			string model =  JoinPath(gSBMLModelsPath, gModels[i]);
			CHECK(gRR->loadSBMLFromFile(model));
        }
	}
}


