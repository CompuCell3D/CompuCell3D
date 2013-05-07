#include <string>
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrIniFile.h"
#include "UnitTest++.h"
#include "rrc_api.h"
#include "src/TestUtils.h"

//using..
using namespace std;
using namespace UnitTest;
using namespace ls;
using namespace rrc;

using rr::JoinPath;
using rr::FileExists;

extern string   gTempFolder;
extern string 	gTestDataFolder;
extern string   gRRInstallFolder;
extern bool 	gDebug;

//This tests is mimicking the Python tests
SUITE(TEST_MODEL_1)
{
string TestDataFileName 	= "Test_1.dat";
IniFile iniFile;
string TestModelFileName;
RRHandle gRR = NULL;

	TEST(DATA_FILES)
	{
    	gRR 						= createRRInstanceE(gTempFolder.c_str());
		string testDataFileName 	= JoinPath(gTestDataFolder, TestDataFileName);
		clog<<"Checking file: "<<testDataFileName<<endl;
		CHECK(FileExists(testDataFileName));
		CHECK(iniFile.Load(testDataFileName));

		clog<<"Loaded test data from file: "<< testDataFileName;
		if(iniFile.GetSection("SBML_FILES"))
		{
			rrIniSection* sbml = iniFile.GetSection("SBML_FILES");
			rrIniKey* fNameKey = sbml->GetKey("FNAME1");
			if(fNameKey)
			{
				TestModelFileName  = JoinPath(gTestDataFolder, fNameKey->mValue);
				CHECK(FileExists(TestModelFileName));
			}
		}
		CHECK(loadSBMLFromFileE(gRR, TestModelFileName.c_str(), true));
	}

	TEST(SET_COMPUTE_AND_ASSIGN_CONSERVATION_LAWS)
	{
		CHECK(gRR!=NULL);
		bool res = setComputeAndAssignConservationLaws(gRR, true);
		CHECK(res);
		clog<<"\nConversation laws: "<<res<<endl;
	}

	TEST(SET_STEADY_STATE_SELECTION_LIST)
	{
		CHECK(gRR!=NULL);

		//Load reference data
		rrIniSection* aSection = iniFile.GetSection("Set Steady State Selection List");
		if(!aSection || !gRR)
		{
			CHECK(false);
			return;
		}

		rrIniKey* aKey = aSection->GetKey("list");
		if(!aKey)
		{
			CHECK(false);
			return;
		}

		bool res = setSteadyStateSelectionList(gRR, aKey->mValue.c_str());
		CHECK(res);
	}

	TEST(GET_STEADY_STATE_SELECTION_LIST)
	{
		rrIniSection* aSection = iniFile.GetSection("Get Steady State Selection List");
		if(!aSection || !gRR)
		{
			CHECK(false);
			return;
		}

		rrIniKey* aKey = aSection->GetKey("list");
		if(!aKey)
		{
			CHECK(false);
			return;
		}

		RRStringArrayHandle list = getSteadyStateSelectionList(gRR);

		if(!list)
		{
			CHECK(false);
			return;
		}
		vector<string> selList = SplitString(aKey->mValue," ,");
		CHECK(selList.size() == list->Count);
		for(int i = 0; i < selList.size(); i++)
		{
			CHECK(selList[i] == list->String[i]);
		}
	}

	TEST(COMPUTE_STEADY_STATE)
	{
		double val;
		CHECK( steadyState(gRR, &val));
		CHECK_CLOSE(0, val, 1e-6);
	}

	TEST(SPECIES_CONCENTRATIONS)
	{
		CHECK(gRR!=NULL);

		//Read in the reference data, from the ini file
		rrIniSection* aSection = iniFile.GetSection("Species Concentrations");
		if(!aSection || !gRR)
		{
			CHECK(false);
			return;
		}

		for(int i = 0 ; i < aSection->KeyCount(); i++)
		{
			rrIniKey *aKey = aSection->GetKey(i);

			double val;

			//API Function getValue
			if(!getValue(gRR, aKey->mKey.c_str(), &val))
			{
				CHECK(false);
			}

			//Check concentrations
			CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
			clog<<"\n";
			clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
		}
	}

    TEST(FLUXES)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Fluxes");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);
            double val;

            //API Function getValue
            if(!getValue(gRR, aKey->mKey.c_str(), &val))
            {
                CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
    }

    TEST(FULL_JACOBIAN)
    {
        rrIniSection* aSection = iniFile.GetSection("Full Jacobian");
        if(!aSection)
        {
            CHECK(false);
            return;
        }

        RRMatrixHandle 		jActual 	= getFullJacobian(gRR);
        ls::DoubleMatrix 	jRef 		= ParseFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(jActual->RSize != jRef.RSize() || jActual->CSize != jRef.CSize())
        {
            CHECK(false);
            return;
        }

        for(int row = 0; row < jActual->RSize; row++)
        {
            for(int col = 0; col < jActual->CSize; col++)
            {
              CHECK_CLOSE(jRef(row,col), jActual->Data[row*jActual->CSize + col], 1e-6);
            }
        }
        //Clean up...
        freeMatrix(jActual);
    }

    TEST(INDIVIDUAL_EIGENVALUES)
    {
        rrIniSection* aSection = iniFile.GetSection("Individual EigenValues");
        if(!aSection)
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);
            clog<<"\n";
            clog<<"Ref_EigenValue: "<<aKey->mKey<<": "<<aKey->mValue<<endl;

            double val;

              string eigenValueLabel ="eigen_" + aKey->mKey;
            if(!getValue(gRR, eigenValueLabel.c_str(), &val))
            {
                CHECK(false);
            }

            clog<<"EigenValue "<<i<<": "<<val<<endl;
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
        }
    }

    TEST(GET_EIGENVALUE_MATRIX)
    {
        CHECK(gRR!=NULL);

        setComputeAndAssignConservationLaws(gRR, true);

        rrIniSection* aSection = iniFile.GetSection("Eigenvalue Matrix");

        //Read in the reference data, from the ini file
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref = ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getEigenvalues(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);


    }


    TEST(STOICHIOMETRY_MATRIX)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Stoichiometry Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

		ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getStoichiometryMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);
        }
        freeMatrix(matrix);
      }

    TEST(LINK_MATRIX)
    {
        clog<<"\n==== LINK_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Link Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

       	ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getLinkMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);
        }
        freeMatrix(matrix);
      }

    TEST(UNSCALED_ELASTICITY_MATRIX)
    {
        clog<<"\n==== UNSCALED_ELASTICITY_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Unscaled Elasticity Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

       	ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getUnscaledElasticityMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);
        }
        freeMatrix(matrix);
      }

    TEST(SCALED_ELASTICITY_MATRIX)
    {
        clog<<"\n==== SCALED_ELASTICITY_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Scaled Elasticity Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        double test;
        steadyState(gRR, &test);
        RRMatrixHandle matrix = getScaledElasticityMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);
      }

    TEST(UNSCALED_CONCENTRATION_CONTROL_MATRIX)
    {
    	clog<<"\n==== UNSCALED_CONCENTRATION_CONTROL_MATRIX ====\n\n";

        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("UnScaled Concentration Control Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getUnscaledConcentrationControlCoefficientMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

        clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

        clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);
    }

	TEST(SCALED_CONCENTRATION_CONTROL_MATRIX)
    {
        clog<<"\n==== SCALED_CONCENTRATION_CONTROL_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Scaled Concentration Control Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getScaledConcentrationControlCoefficientMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);
      }

    TEST(UNSCALED_FLUX_CONTROL_MATRIX)
    {
    	clog<<"\n==== UNSCALED_FLUX_CONTROL_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Unscaled Flux Control Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getUnscaledFluxControlCoefficientMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);
      }

	TEST(SCALED_FLUX_CONTROL_MATRIX)
    {
    	clog<<"\n==== SCALED_FLUX_CONTROL_MATRIX ====\n\n";
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Scaled Flux Control Matrix");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        ls::DoubleMatrix 	ref 		= ParseFromText(aSection->GetNonKeysAsString());

        RRMatrixHandle matrix = getScaledFluxControlCoefficientMatrix(gRR);
        if(!matrix)
        {
            CHECK(false);
            return;
        }

          clog<<"Reference Matrix:\n";
        clog<<ref<<endl;

          clog<<"Calculated Matrix:";
        clog<<matrixToString(matrix);

        if(!matrix || matrix->RSize != ref.RSize())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < ref.RSize(); i++)
        {
            double val;
            if(!getMatrixElement(matrix, i , 0, &val))
            {
                CHECK(false);
            }
            CHECK_CLOSE(ref(i,0), val, 1e-6);

        }
        freeMatrix(matrix);
      }

    TEST(FLOATING_SPECIES_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Floating Species Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getFloatingSpeciesIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(BOUNDARY_SPECIES_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Boundary Species Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getBoundarySpeciesIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(GLOBAL_PARAMETER_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Global Parameter Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getGlobalParameterIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(COMPARTMENT_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Compartment Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getCompartmentIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(REACTION_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Reaction Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getReactionIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(SPECIES_INITIAL_CONDITION_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Species Initial Condition Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getFloatingSpeciesInitialConditionIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(GET_EIGENVALUE_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Get Eigenvalue Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getEigenvalueIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(GET_RATES_OF_CHANGE_IDS)
    {
        rrIniSection* aSection = iniFile.GetSection("Get Rates Of Change Ids");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getRatesOfChangeIds(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(SET_STEADY_STATE_SELECTION_LIST_2)
    {
        CHECK(gRR!=NULL);

        //Load reference data
        rrIniSection* aSection = iniFile.GetSection("Set Steady State Selection List 2");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        bool res = setSteadyStateSelectionList(gRR, aKey->mValue.c_str());
        CHECK(res);
    }

    TEST(GET_STEADY_STATE_SELECTION_LIST_2)
    {
        rrIniSection* aSection = iniFile.GetSection("Get Steady State Selection List 2");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getSteadyStateSelectionList(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(SET_TIME_COURSE_SELECTION_LIST)
    {
        CHECK(gRR!=NULL);

        //Load reference data
        rrIniSection* aSection = iniFile.GetSection("Set Time Course Selection List");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        bool res = setTimeCourseSelectionList(gRR, aKey->mValue.c_str());
        CHECK(res);
    }

    TEST(GET_TIME_COURSE_SELECTION_LIST)
    {
        rrIniSection* aSection = iniFile.GetSection("Get Time Course Selection List");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRStringArrayHandle list = getTimeCourseSelectionList(gRR);

        if(!list)
        {
            CHECK(false);
            return;
        }
        vector<string> selList = SplitString(aKey->mValue," ,");
        CHECK(selList.size() == list->Count);
        for(int i = 0; i < selList.size(); i++)
        {
            CHECK(selList[i] == list->String[i]);
        }
    }

    TEST(COMPUTE_STEADY_STATE_VALUES)
    {
        rrIniSection* aSection = iniFile.GetSection("Compute Steady State Values");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        RRVector* values = computeSteadyStateValues(gRR);

        if(!values || values->Count != aSection->KeyCount())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(FLOATING_SPECIES_CONCENTRATIONS)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Floating Species Concentrations");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getFloatingSpeciesConcentrations(gRR);

        if(!values || values->Count != aSection->KeyCount())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

        TEST(BOUNDARY_SPECIES_CONCENTRATIONS)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Boundary Species Concentrations");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getBoundarySpeciesConcentrations(gRR);

        if(!values || values->Count != aSection->KeyCount())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_GLOBABL_PARAMETER_VALUES)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Global Parameter Values");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getGlobalParameterValues(gRR);

        vector<string> refList = SplitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_INITIAL_FLOATING_SPECIES_CONCENTRATIONS)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Initial Floating Species Concs");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getFloatingSpeciesInitialConcentrations(gRR);

        vector<string> refList = SplitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_REACTION_RATES)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Reaction Rates");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getReactionRates(gRR);

        vector<string> refList = SplitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {

            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_REACTION_RATE_BY_INDEX)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Reaction Rates");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        vector<string> refList = SplitString(aKey->mValue," ,");

        if(refList.size() != getNumberOfReactions(gRR))
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            double value;
              if(!getReactionRate(gRR, i, &value))
              {
                CHECK(false);
              }

            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), value, 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<value<<endl;
        }
    }

    TEST(NUMBER_OF_DEPENDENT_SPECIES)
    {
        rrIniSection* aSection = iniFile.GetSection("Number of Dependent Species");
        if(!aSection)
        {
            CHECK(false);
            return;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          rrIniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfDependentSpecies(gRR);

          CHECK(aKey->AsInt() ==  val);
    }

    TEST(NUMBER_OF_INDEPENDENT_SPECIES)
    {
        rrIniSection* aSection = iniFile.GetSection("Number of Independent Species");
        if(!aSection)
        {
            CHECK(false);
            return;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          rrIniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfIndependentSpecies(gRR);

          CHECK(aKey->AsInt() ==  val);
    }

        TEST(NUMBER_OF_RULES)
    {
        rrIniSection* aSection = iniFile.GetSection("Number of Rules");
        if(!aSection)
        {
            CHECK(false);
            return;
        }

        if(aSection->KeyCount() < 1)
        {
            CHECK(false);
          }

          rrIniKey *aKey = aSection->GetKey(0);
          int val = getNumberOfRules(gRR);

          CHECK(aKey->AsInt() ==  val);
    }

	TEST(GET_RATES_OF_CHANGE)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Rates Of Change");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* aKey = aSection->GetKey("list");
        if(!aKey)
        {
            CHECK(false);
            return;
        }

        RRVector* values = getRatesOfChange(gRR);

        vector<string> refList = SplitString(aKey->mValue," ,");

        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_REACTION_RATES_EX)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Reaction Rates Ex");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        rrIniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = SplitString(refs->mValue," ");

          vector<string> concList = SplitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = ToDouble(concList[i]);
          }

        RRVector* values = getReactionRatesEx(gRR, aVector);


        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_RATES_OF_CHANGE_EX)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Rates Of Change Ex");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        rrIniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = SplitString(refs->mValue," ");

          vector<string> concList = SplitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = ToDouble(concList[i]);
          }

        RRVector* values = getRatesOfChangeEx(gRR, aVector);


        if(!values || values->Count != refList.size())
        {
            CHECK(false);
            return;
        }

        for(int i = 0 ; i < refList.size(); i++)
        {
            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), values->Data[i], 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<values->Data[i]<<endl;
        }
    }

    TEST(GET_RATE_OF_CHANGE_BY_INDEX)
    {
        CHECK(gRR!=NULL);

        //Read in the reference data, from the ini file
        rrIniSection* aSection = iniFile.GetSection("Get Rates Of Change Ex");
        if(!aSection || !gRR)
        {
            CHECK(false);
            return;
        }

        rrIniKey* conc = aSection->GetKey("list1");
        if(!conc)
        {
            CHECK(false);
            return;
        }

        rrIniKey* refs = aSection->GetKey("list2");
        if(!refs)
        {
            CHECK(false);
            return;
        }

          vector<string> refList = SplitString(refs->mValue," ");

          vector<string> concList = SplitString(conc->mValue," ");
          RRVector* aVector = createVector(concList.size());

          for(int i = 0; i < concList.size(); i++)
          {
            aVector->Data[i] = ToDouble(concList[i]);
          }

        for(int i = 0 ; i < refList.size(); i++)
        {
            double value;
              if(!getRateOfChange(gRR, i, &value))
              {
                CHECK(false);
              }

            //Check concentrations
            CHECK_CLOSE(ToDouble(refList[i]), value, 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<ToDouble(refList[i])<<"\tActual:\t "<<value<<endl;
        }
    }

    TEST(FREE_RR_INSTANCE)
    {
        CHECK(freeRRInstance(gRR));
    }

}
