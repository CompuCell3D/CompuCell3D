#include <complex>
#include "unit_test/UnitTest++.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrIniFile.h"
#include "rrTestUtils.h"

using namespace UnitTest;
using namespace rr;

extern RoadRunner* 		gRR;
extern string 			gTestDataFolder;
extern string 			gCompiler;
extern string 			gSupportCodeFolder;
extern string 			gTempFolder;
extern string		 	gRRInstallFolder;

//Global to this unit
RoadRunner *aRR = NULL;
string TestDataFileName 	= "ss_ThreeSpecies.dat";
string TestModelFileName    = "";
IniFile iniFile;

SUITE(ssThreeSpecies)
{
	//This suite tests various steady state functions, using the model TestModel_1.xml

	//Test that model files and reference data for the tests in this suite are present
    TEST(DATA_FILES)
    {
		gTestDataFolder = JoinPath(gRRInstallFolder, "tests");
		TestDataFileName 	= JoinPath(gTestDataFolder, TestDataFileName);

    	CHECK(FileExists(TestDataFileName));
        CHECK(iniFile.Load(TestDataFileName));
        clog<<"Loaded test data from file: "<< TestDataFileName;
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

	TEST(LOAD_MODEL)
	{
		aRR = new RoadRunner(gSupportCodeFolder, gCompiler, gTempFolder);
		CHECK(aRR!=NULL);

        //Load the model
        aRR->computeAndAssignConservationLaws(true);
		CHECK(aRR->loadSBMLFromFile(TestModelFileName));
	}

    TEST(COMPUTE_STEADY_STATE)
    {
        //Compute Steady state
		if(aRR)
        {
	        CHECK_CLOSE(0, aRR->steadyState(), 1e-6);
        }
        else
        {
        	CHECK(false);
        }
    }

    TEST(STEADY_STATE_CONCENTRATIONS)
	{
        rrIniSection* aSection = iniFile.GetSection("STEADY_STATE_CONCENTRATIONS");
        //Read in the reference data, from the ini file
		if(!aSection || !aRR)
        {
        	CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
            rrIniKey *aKey = aSection->GetKey(i);
            double val = aRR->getValue(aKey->mKey);

            //Check concentrations
            CHECK_CLOSE(aKey->AsFloat(), val, 1e-6);
        }
    }

    TEST(FULL_JACOBIAN)
	{
		rrIniSection* aSection = iniFile.GetSection("FULL_JACOBIAN");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        DoubleMatrix fullJacobian 	= aRR->getFullJacobian();
        DoubleMatrix jRef 			= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
        {
        	CHECK(false);
            return;
        }

        clog<<fullJacobian;
		CHECK_ARRAY2D_CLOSE(jRef, fullJacobian, fullJacobian.RSize(),fullJacobian.CSize(), 1e-6);
    }

    TEST(REDUCED_JACOBIAN)
	{
		rrIniSection* aSection = iniFile.GetSection("REDUCED_JACOBIAN");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        DoubleMatrix fullJacobian 	= aRR->getReducedJacobian();
        DoubleMatrix jRef 			= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(fullJacobian.RSize() != jRef.RSize() || fullJacobian.CSize() != jRef.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(jRef, fullJacobian, fullJacobian.RSize(),fullJacobian.CSize(), 1e-6);
    }

    TEST(FULL_REORDERED_JACOBIAN)
	{
		rrIniSection* aSection = iniFile.GetSection("FULL_REORDERED_JACOBIAN");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix = aRR->getFullReorderedJacobian();
        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

		cout<<"Reference\n"<<ref;
		cout<<"matrix\n"<<matrix;

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }

    TEST(EIGEN_VALUES)
	{
        rrIniSection* aSection = iniFile.GetSection("EIGEN_VALUES");
        //Read in the reference data, from the ini file
		if(!aSection || !aRR)
        {
        	CHECK(false);
            return;
        }

        vector<Complex> eigenVals = aRR->getEigenvaluesCpx();
        if(eigenVals.size() != aSection->KeyCount())
        {
        	CHECK(false);
            return;
        }

        for(int i = 0 ; i < aSection->KeyCount(); i++)
        {
        	clog<<"EigenValue "<<i<<"_ref: "<<aSection->GetKey(i)->AsString()<<endl;
            rrIniKey *aKey = aSection->GetKey(i);
            std::complex<double> eig(aKey->AsComplex());
        	clog<<"EigenValue "<<i<<": "<<real(eigenVals[i])<<endl;
            CHECK_CLOSE(aKey->AsFloat(), real(eigenVals[i]), 1e-6);
        }
    }

    TEST(STOICHIOMETRY_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("STOICHIOMETRY_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix mat = aRR->getStoichiometryMatrix();
        DoubleMatrix ref = ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
    }

    TEST(REORDERED_STOICHIOMETRY_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("REORDERED_STOICHIOMETRY_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix mat		 	= aRR->getReorderedStoichiometryMatrix();
        DoubleMatrix ref 			= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
    }

    TEST(FULLY_REORDERED_STOICHIOMETRY_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("FULLY_REORDERED_STOICHIOMETRY_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix mat		 	= aRR->getFullyReorderedStoichiometryMatrix();
        DoubleMatrix ref 			= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(mat.RSize() != ref.RSize() || mat.CSize() != ref.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, mat, mat.RSize(), mat.CSize(), 1e-6);
    }

    TEST(LINK_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("LINK_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix 	= *(aRR->getLinkMatrix());
        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(false);
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }

    TEST(UNSCALED_ELASTICITY_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("UNSCALED_ELASTICITY_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
		DoubleMatrix matrix 	= aRR->getUnscaledElasticityMatrix();
        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(!"Wrong matrix dimensions" );
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }

    TEST(SCALED_ELASTICITY_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("SCALED_ELASTICITY_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
		DoubleMatrix matrix 	= aRR->getScaledReorderedElasticityMatrix();
        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(!"Wrong matrix dimensions" );
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }

    TEST(UNSCALED_CONCENTRATION_CONTROL_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("UNSCALED_CONCENTRATION_CONTROL_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix 	= aRR->getUnscaledConcentrationControlCoefficientMatrix();
        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(!"Wrong matrix dimensions" );
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }

    TEST(UNSCALED_FLUX_CONTROL_MATRIX)
	{
		rrIniSection* aSection = iniFile.GetSection("UNSCALED_FLUX_CONTROL_MATRIX");
   		if(!aSection)
        {
        	CHECK(false);
            return;
        }

        //Read in the reference data, from the ini file
        DoubleMatrix matrix 	= aRR->getUnscaledFluxControlCoefficientMatrix();
        DoubleMatrix ref  		= ParseMatrixFromText(aSection->GetNonKeysAsString());

        //Check dimensions
        if(matrix.RSize() != ref.RSize() || matrix.CSize() != ref.CSize())
        {
        	CHECK(!"Wrong matrix dimensions" );
            return;
        }

		CHECK_ARRAY2D_CLOSE(ref, matrix, matrix.RSize(), matrix.CSize(), 1e-6);
    }
}

