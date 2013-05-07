#include <string>
#include "UnitTest++.h"
#include "rrc_api.h"
#include "rrUtils.h"
#include "rrIniFile.h"
#include "rrException.h"
#include "TestUtils.h"
using namespace std;
using namespace UnitTest;

//Add using clauses..
using rr::JoinPath;
using rr::FileExists;

extern RRHandle gRR;	//Global roadrunner C handle
extern string 	gBinPath;
extern string 	gSBMLModelsPath;
extern string 	gCompilerPath;
extern string 	gSupportCodeFolder;
extern string 	gTestDataFolder;
extern string		 	gRRInstallFolder;
SUITE(SteadyState)
{

string TestDataFileName 	= "TestModel_1.dat";
IniFile iniFile;
string TestModelFileName;

	//Test that model files and reference data for the tests in this suite are present
	TEST(DATA_FILES)
	{
		gTestDataFolder 			= JoinPath(gRRInstallFolder, "tests");
		string testDataFileName 	= JoinPath(gTestDataFolder, TestDataFileName);

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
	}

    TEST(AllocateRR)
    {
        if(!gRR)
        {
            gRR = getRRInstance();
        }
        CHECK(gRR!=NULL);
        freeRRInstance(gRR);
        gRR = NULL;
        CHECK(gRR == NULL);
    }


	TEST(LOAD_MODEL)
	{
     	gRR = getRRInstance();
        CHECK(gRR!=NULL);

        //Load the model
        setComputeAndAssignConservationLaws(true);
		CHECK(loadSBMLFromFile(TestModelFileName.c_str()));
	}

    TEST(COMPUTE_STEADY_STATE)
    {
        //Compute Steady state
        double val;
        CHECK( steadyState(val));
        CHECK_CLOSE(0, val, 1e-6);
    }

//	TEST(GET_UCC)
//	{
//		double val;
//		getuCC("S1","S2", val);
//	}

	TEST(GET_EIGENVALUE_IDS)
	{
		gRR = getRRInstance();
        if(!gRR)
        {
        	CHECK(false);
        	return;
        }

        //Load reference data
        rrIniSection* aSection = iniFile.GetSection("GET_EIGENVALUE_IDS");
		if(!aSection || !gRR)
        {
        	CHECK(false);
            return;
        }

        //Load the model
        CHECK(loadSBMLFromFile(TestModelFileName.c_str()));

        rrIniKey* aKey = aSection->GetKey("COMPUTE_AND_ASSIGN_CONSERVATION_LAWS");

        if(aKey)
        {
        	setComputeAndAssignConservationLaws(aKey->AsBool());
        }
        else
        {
			CHECK(aKey != NULL);
        }

        for(int i = 1 ; i < aSection->KeyCount(); i++) //OBS, ignore first key, which is computeAndAssignConservation Laws
        {
            rrIniKey *aKey = aSection->GetKey(i);
            RRStringArrayHandle ids = getEigenvalueIds();

            if(ids != NULL && i < (ids->Count + 1))
            {
            	CHECK(strcmp(aKey->mValue.c_str(),ids->String[i - 1]) == 0);
            }
            else
            {
            	CHECK(false);
            }
        }

        //Test to call many times ,
        //Todo: add a simple way to look for memory leak later on..
        for(int i = 0; i < 10000; i++)
        {
        	RRStringArrayHandle ids = getEigenvalueIds();
        	CHECK(getEigenvalueIds());
            freeStringArray(ids);
        }
    }

    TEST(STEADY_STATE_CONCENTRATIONS)
	{
       	gRR = getRRInstance();
        CHECK(gRR!=NULL);
        rrIniSection* aSection = iniFile.GetSection("STEADY_STATE_CONCENTRATIONS");
        //Read in the reference data, from the ini file
		if(!aSection || !gRR)
        {
        	CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);
            double val;
            if(!getValue(aKey->mKey.c_str(), val))
            {
            	CHECK(false);
            }

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
            clog<<"\n";
            clog<<"Ref:\t"<<aKey->AsFloat()<<"\tActual:\t "<<val<<endl;
        }
    }

    //This test is using the function getValue("eigen_...")
    //The test is checking for proper exception when Conservation laws are not enabled
    TEST(GET_EIGENVALUES_THROW_EXCEPTION)
	{
       	gRR = getRRInstance();
        CHECK(gRR!=NULL);
        setComputeAndAssignConservationLaws(false);
        rrIniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
		if(!aSection || !gRR)
        {
        	CHECK(false);
            return;
        }

        RRStringArrayHandle ids = getEigenvalueIds();
        if(!ids)
        {
        	CHECK(false);
            return;
        }
        if(ids->Count != aSection->KeyCount() && ids->Count < 1)
        {
        	CHECK(false);
            return;
        }

        double val;
        bool res = getValue(ids->String[0], val);

        if(getLastError())
        {
        	char* error = getLastError();
        	clog<<endl<<error;
        }
        CHECK(res == false);
    }

	//This test is using the function getValue("eigen_...")
    TEST(GET_EIGENVALUES_1)
	{
       	gRR = getRRInstance();
        CHECK(gRR!=NULL);
        setComputeAndAssignConservationLaws(true);
        rrIniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
        //Read in the reference data, from the ini file
		if(!aSection || !gRR)
        {
        	CHECK(false);
            return;
        }

        RRStringArrayHandle ids = getEigenvalueIds();
        if(!ids)
        {
        	CHECK(false);
			return;
        }
        if(ids->Count != aSection->KeyCount())
        {
        	CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
        	//Find correct eigenValue
            for(int j = 0; j < ids->Count; j++)
            {
            	if(aSection->mKeys[i]->mKey == ids->String[j])
                {
                    rrIniKey *aKey = aSection->GetKey(i);
                    clog<<"\n";
                    clog<<"Ref_EigenValue: "<<aKey->mKey<<": "<<aKey->mValue<<endl;

                    double val;
                    if(!getValue(ids->String[j], val))
                    {
                    	CHECK(false);
                        continue;
                    }
                    clog<<"ID: "<<ids->String[j]<<"= "<<val<<endl;

                    CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
                }
            }
        }
    }

	//Using getEigenValues
    TEST(GET_EIGENVALUES_2)
	{
       	gRR = getRRInstance();
        CHECK(gRR!=NULL);
        setComputeAndAssignConservationLaws(true);

        rrIniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
        //Read in the reference data, from the ini file
		if(!aSection || !gRR)
        {
        	CHECK(false);
            return;
        }

        RRMatrixHandle eigenVals = getEigenvalues();
		if(!eigenVals)
		{
			CHECK(false);
			return;
		}
        clog<<matrixToString(eigenVals);
        if(!eigenVals || eigenVals->RSize != aSection->KeyCount())
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
            if(!getMatrixElement(eigenVals, i , 0, val))
            {
            	CHECK(false);
            }
        	clog<<"EigenValue "<<i<<": "<<val<<endl;
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);

        }
		freeMatrix(eigenVals);
    }

	//Using getEigenValues
    TEST(COMPUTE_STEADY_STATE_VALUES)
	{
       	gRR = getRRInstance();
        CHECK(gRR!=NULL);
        setComputeAndAssignConservationLaws(true);
        RRVector* vec = computeSteadyStateValues();
        clog<<vectorToString(vec)<<endl;
    }

    TEST(FULL_JACOBIAN)
	{
		rrIniSection* aSection = iniFile.GetSection("FULL_JACOBIAN");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        RRMatrixHandle 		jActual 	= getFullJacobian();
        ls::DoubleMatrix 	jRef 		= ParseFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(jActual->RSize != jRef.RSize() || jActual->CSize != jRef.CSize())
        {
        	CHECK(false);
            return;
        }

        double *mat = jActual->Data;

        for(int row = 0; row < jActual->RSize; row++)
        {
            for(int col = 0; col < jActual->CSize; col++)
            {
              CHECK_CLOSE(jRef(row,col), jActual->Data[row*jActual->CSize + col], 1e-6);
            }
        }


        //Check matrixToString
        char* strMat = matrixToString(jActual);
		clog<<strMat;

		//Clean up...
        freeMatrix(jActual);

        freeText(strMat);
    }

//    TEST(REDUCED_JACOBIAN)
//	{
//		rrIniSection* aSection = iniFile.GetSection("REDUCED_REORDERED_JACOBIAN");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        DoubleMatrix jActual 	= aRR->getReducedJacobian();
//        DoubleMatrix jRef 			= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(jActual.RSize() != jRef.RSize() || jActual.CSize() != jRef.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//        clog<<"Reduced Jacobian\n"<<jActual;
//		CHECK_ARRAY2D_CLOSE(jRef, jActual, jActual.RSize(),jActual.CSize(), 1e-6);
//    }
//
//    TEST(FULL_REORDERED_JACOBIAN)
//	{
//		rrIniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix matrix = aRR->getFullReorderedJacobian();
//        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//		cout<<"Reference\n"<<ref;
//		cout<<"matrix\n"<<matrix;
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(REDUCED_REORDERED_JACOBIAN)
//	{
//		rrIniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix matrix = aRR->getReducedJacobian();
//        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//		cout<<"Reference\n"<<ref;
//		cout<<"matrix\n"<<matrix;
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(STOICHIOMETRY_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("STOICHIOMETRY_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix mat = aRR->getStoichiometryMatrix();
//        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
//    }
//
//    TEST(REORDERED_STOICHIOMETRY_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("REORDERED_STOICHIOMETRY_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix mat		 	= aRR->getReorderedStoichiometryMatrix();
//        DoubleMatrix ref 			= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
//    }
//
//    TEST(FULLY_REORDERED_STOICHIOMETRY_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("FULLY_REORDERED_STOICHIOMETRY_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix mat		 	= aRR->getFullyReorderedStoichiometryMatrix();
//        DoubleMatrix ref 			= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
//    }
//
//    TEST(LINK_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("LINK_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix matrix 	= *(aRR->getLinkMatrix());
//        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(false);
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(UNSCALED_ELASTICITY_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("UNSCALED_ELASTICITY_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//		DoubleMatrix matrix 	= aRR->getUnscaledElasticityMatrix();
//        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(!"Wrong matrix dimensions" );
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(SCALED_ELASTICITY_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("SCALED_ELASTICITY_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//		DoubleMatrix matrix 	= aRR->getScaledReorderedElasticityMatrix();
//        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(!"Wrong matrix dimensions" );
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(UNSCALED_CONCENTRATION_CONTROL_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("UNSCALED_CONCENTRATION_CONTROL_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix matrix 	= aRR->getUnscaledConcentrationControlCoefficientMatrix();
//        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(!"Wrong matrix dimensions" );
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }
//
//    TEST(UNSCALED_FLUX_CONTROL_MATRIX)
//	{
//		rrIniSection* aSection = iniFile.GetSection("UNSCALED_FLUX_CONTROL_MATRIX");
//   		if(!aSection)
//        {
//        	CHECK(false);
//            return;
//        }
//
//        //Read in the reference data, from the ini file
//        DoubleMatrix matrix 	= aRR->getUnscaledFluxControlCoefficientMatrix();
//        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());
//
//        //Check dimensions
//        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
//        {
//        	CHECK(!"Wrong matrix dimensions" );
//            return;
//        }
//
//		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
//    }


}//SUITE


