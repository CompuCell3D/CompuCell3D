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
extern string 			gTempFolder;
extern vector<string> 	gModels;
SUITE(Stoichiometric)
{
	TEST(AllocateRR)
	{
		if(!gRR)
		{
			gRR = new RoadRunner(gSupportCodeFolder, gCompiler, gTempFolder);
		}

		CHECK(gRR!=NULL);
	}

    TEST(MODEL_FILES)	//Test that model files for the tests are present
    {
    }
}


