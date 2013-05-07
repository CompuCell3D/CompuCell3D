#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "rrc_api.h"
#include "rrc_support.h"
//---------------------------------------------------------------------------

using namespace std;
using namespace rrc;;

void printMatrix(char* msg1, RRMatrixHandle mat);
int main(int argc, char* argv[])
{
	enableLogging();
    setLogLevel("Debug3");

	string modelsPath(".\\..\\Models");
	if(argc > 1)
	{
		modelsPath = argv[1];
	}

	RRHandle rrHandle = NULL;
    rrHandle =  getRRInstance();

    if(!rrHandle)
    {
        cout<<"No handle...";
    }

	cout<<"API Version: "<<getVersion()<<endl;
	cout<<"libSBML Version: "<<getlibSBMLVersion()<<endl;
    setTempFolder("c:\\rrTemp");
    enableLogging();

	text = getLogFileName();
    if(text)
	{
		cout<<"Log File Name: "<<text<<endl;
		freeText(text);
	}

	text = getBuildDate();

	if(text)
	{
		cout<<"Build date: "<<text<<endl;
		freeText(text);
	}

//	   string fileName = modelsPath + "\\ss_TurnOnConservationAnalysis.xml";
//	   string fileName = modelsPath + "\\ss_SimpleConservedCycle.xml";
	 string fileName = modelsPath + "\\ss_threeSpecies.xml";
//	 string fileName = modelsPath + "\\selectionListBug.xml";
//	 string fileName = modelsPath + "\\boundary.xml";

	ifstream ifs(fileName.c_str());
	if(!ifs)
	{
		cerr<<"Failed opening file: "<<fileName;
		return false;
	}
	cout << "\nRunning model: " << fileName << endl;
	setComputeAndAssignConservationLaws(false);
	std::string sbml((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

	if(!loadSBML(sbml.c_str()))
	{
		cerr<<"Failed loading SBML from file:"<<fileName<<endl;
		cerr<<"Last error was: "<<getLastError()<<endl;
		return -1;
	}

	RRListHandle sList = getAvailableTimeCourseSymbols();
    cout<<"Symbols: "<<listToString(sList);

    sList = getAvailableSteadyStateSymbols();
    cout<<"\n\n Steady state symbols\n";
    cout<<listToString(sList);
    freeRRList(sList);

    char* cFileName = getCSourceFileName();
    if(cFileName)
    {
    	cout<<"\n C File Name: "<<cFileName<<"\n";
    }

    freeText(cFileName);

    cout<<"Number of rules: "<<getNumberOfRules()<<"\n";
	int r = getNumberOfReactions();
	int m = getNumberOfFloatingSpecies();
	int b = getNumberOfBoundarySpecies();
	int p = getNumberOfGlobalParameters();
	int c = getNumberOfCompartments();

	printf ("Number of reactions = %d\n", r);
	printf ("Number of floating species = %d\n", m);
	printf ("Number of boundary species = %d\n\n", b);
	printf ("Number of compartments = %d\n\n", c);

	if (m > 0) {
	   printf ("Compartment names:\n");
	   printf ("------------------\n");
	   cout<<stringArrayToString(getCompartmentIds())<<endl<<endl;
	}

	if (m > 0) {
	   printf ("Floating species names:\n");
	   printf ("-----------------------\n");
	   cout<<stringArrayToString(getFloatingSpeciesIds())<<endl<<endl;
	}

	if (m > 0) {
	   printf ("Initial Floating species names:\n");
	   printf ("-------------------------------\n");
	   cout<<stringArrayToString(getFloatingSpeciesInitialConditionIds())<<endl;
	}

	if (b > 0) {
       printf ("\nBoundary species names:\n");
	   printf ("-----------------------\n");
	   cout<<stringArrayToString(getBoundarySpeciesIds())<<endl;
	}
	printf ("\n");

	if (p > 0) {
       printf ("\nGlobal Parameter names:\n");
	   printf ("-----------------------\n");
	   cout<<stringArrayToString(getGlobalParameterIds())<<endl;
	}
	printf ("\n");

	if (r > 0) {
       printf ("\nReaction names:\n");
	   printf ("---------------\n");
	   cout<<stringArrayToString(getReactionIds())<<endl;
	}
	printf ("\n");

	if (m> 0) {
       printf ("\nRates of change names:\n");
	   printf ("----------------------\n");
	   cout<<stringArrayToString(getRatesOfChangeIds())<<endl;
	}
	printf ("\n");


	if (r > 0) {
       printf ("\nUnscaled flux control coefficient names:\n");
	   printf ("----------------------------------------\n");
	   RRListHandle stringArray = getUnscaledFluxControlCoefficientIds();
	   cout<<listToString(stringArray)<<endl;
	}
	printf ("\n");

	if (m > 0) {
       printf ("\nUnscaled concentration control coefficient names:\n");
	   printf ("-------------------------------------------------\n");
	   cout<<listToString(getUnscaledConcentrationControlCoefficientIds())<<endl;
	}
	printf ("\n");

	double ssVal;
    bool success = steadyState(ssVal);
    if(!success)
    {
		cerr<<"Steady State call failed. Error was: "<<getLastError()<<endl;
    }
    else
    {
	    cout<<"Compute Steady State: sums of squares: "<<ssVal<<endl;
    }

	cout<<"Steady State selection List: "<<listToString(getSteadyStateSelectionList());
	setSteadyStateSelectionList("S2 S1");
	cout<<"\nSteady State selection List: "<<listToString(getSteadyStateSelectionList());

    printMatrix("Stoichiometry Matrix", getStoichiometryMatrix());

    cout<<"Number of independent species = "<<getNumberOfIndependentSpecies()<<endl;
    cout<<"Number of dependent Species = "<<getNumberOfDependentSpecies()<<endl<<endl;

    printMatrix("Link Matrix", getLinkMatrix());
	printMatrix("Nr Matrix", getNrMatrix());
	printMatrix("L0 Matrix", getL0Matrix());
	printMatrix("Full Jacobian Matrix", getFullJacobian());
	printMatrix("Reduced Jacobian Matrix:", getReducedJacobian());
    printMatrix("Eigenvalue Matrix (real/imag)", getEigenValues());
	printMatrix("Unscaled Elasticity Matrix:", getUnScaledElasticityMatrix());
    printMatrix("Scaled Elasticity Matrix:", getScaledElasticityMatrix());
	printMatrix("Unscaled Concentration Control Coefficients Matrix", getUnscaledConcentrationControlCoefficientMatrix());
	printMatrix("Scaled Concentration Control Coefficients Matrix:", getScaledConcentrationControlCoefficientMatrix());
	printMatrix("Unscaled Flux Control Coefficients Matrix", getUnscaledFluxControlCoefficientMatrix());
	printMatrix("Scaled Flux Control Coefficients Matrix", getScaledFluxControlCoefficientMatrix());

	double value;
	printf ("Flux Control Coefficient, CC^(_J1)_k1\n");
	getCC("_J1", "k1", value);
	printf ("Coefficient = %f\n", value);

	printf ("Flux Control Coefficient, CC^(_J1)_k2\n");
	getCC("_J1", "k2", value);
	printf ("Coefficient = %f\n", value);

	printf ("Flux Control Coefficient, CC^(_J1)_k3\n");
	getCC("_J1", "k3", value);
	printf ("Coefficient = %f\n", value);

	printf ("Flux Control Coefficient, CC^(_J1)_k4\n");
	getCC("_J1", "k4", value);
	printf ("Coefficient = %f\n", value);

	printf ("Elasticity Coefficient, EE^(_J1)_S1\n");
	getEE("_J1", "S1", value);
	printf ("Elasticity = %f\n", value);

	printf ("Elasticity Coefficient, EE^(_J2)_S1\n");
	getEE("_J2", "S1", value);
	printf ("Elasticity = %f\n", value);

	printf ("Elasticity Coefficient, EE^(_J2)_S2\n");
	getEE("_J2", "S2", value);
	printf ("Elasticity = %f\n", value);

	printf ("Elasticity Coefficient, EE^(_J3)_S2\n");
	getEE("_J3", "S2", value);
	printf ("Elasticity = %f\n", value);

	printf ("\n");
	//printf ("Flux Control Coefficient, C^(_J1)_k1\n");
	//double value;
	//getCC("_J1", "k1", value);
	//printf ("FCC = %f\n", value);

	/*getGlobalParameterByIndex (0, value);
	printf ("%f\n", value);
	getGlobalParameterByIndex (1, value);
	printf ("%f\n", value);
	getGlobalParameterByIndex (2, value);
	printf ("%f\n", value);
	getGlobalParameterByIndex (3, value);
	printf ("%f\n", value);*/

	RRVector veca;
	veca.Count = 3;
	veca.Data = new double[3];
   	veca.Data[0] = 1;
	veca.Data[1] = 2;
	veca.Data[2] = 3;

    double aValue = 231.23;
    bool bResult = setVectorElement(&veca, 0, aValue);
    if(!bResult)
    {
    	cout<<"Problem";
    }


    cout<<"List of floating species: \n"<<stringArrayToString(getFloatingSpeciesIds())<<endl;

	printf ("\nCall to getRatesOfChangeEx (S1=1, S2=2, S3=3):\n");
	cout<<vectorToString (getRatesOfChangeEx(&veca))<<endl;

//	printf ("\nCall to getReactionRatesEx (S1=1, S2=2, S3=3):\n");
//	cout<<printVector (getReactionRatesEx (&veca))<<endl;
//
//	printf ("\nCall to getRatesOfChange (with S1=1, S2=2, S3=3):\n");
//	cout<<printVector (getRatesOfChange())<<endl;

    setTimeCourseSelectionList("S1 S2");
//-------- The latest
    cout<<vectorToString(getFloatingSpeciesConcentrations());
    cout<<vectorToString(getGlobalParameterValues());
    cout<<"\n\n Symbols\n";
    RRList* symHandle = getAvailableTimeCourseSymbols();
    cout<<listToString(symHandle);
    freeRRList(symHandle);

    symHandle = getAvailableSteadyStateSymbols();
    cout<<"\n\n Steady state symbols\n";
    cout<<listToString(symHandle);
    freeRRList(symHandle);

    cout<<"\n\n ================================\n";
    RRVector* test = getReactionRates();
    cout<<vectorToString(test);

    setFloatingSpeciesByIndex(0,2);
    setFloatingSpeciesByIndex(1,4);
    setFloatingSpeciesByIndex(2,6);

    test = getReactionRates();
    cout<<vectorToString(test);

    //Get value problem..
    getValue("S1", value);
    cout<<value<<endl;
    getValue("S2", value);
    cout<<value<<endl;
    getValue("S3", value);
    cout<<value<<endl;

    getRatesOfChange();

    getValue("S1", value);
    cout<<value<<endl;
    getValue("S2", value);
    cout<<value<<endl;
    getValue("S3", value);
    cout<<value<<endl;

	//cout<<getBoundarySpeciesByIndex (0)<<endl;
    //getGlobalParameterByIndex(0, value);

    //cout<<value<<endl;
    //getGlobalParameterByIndex(2, value);
    //cout<<value<<endl;

    //cout<<getParamPromotedSBML(sbml.c_str());

    //cout<<getSBML()<<endl;

    //cout<<printMatrix(getScaledElasticityMatrix());     //How to free, when doing something like this??
    //cout<<printStringList(getEigenValueNames());

    cout<<"\n FluxControlCoeff ------\n"<<listToString(getFluxControlCoefficientIds())<<endl;

    cout<<"\n Unscaled FluxControlCoeff ------\n"<<listToString(getUnscaledFluxControlCoefficientIds())<<endl;
    RRList* list =getConcentrationControlCoefficientIds();
    cout<<listToString(list)<<endl;
    //freeList(list);


    //cout<<printStringList(getElasticityNames())<<endl;

//    setBoundarySpeciesByIndex(0,34);
    cout<<"Nr of Compartments: "<<getNumberOfCompartments()<<endl;
    setCompartmentByIndex(0,456);
    if(getCompartmentByIndex(0, value))
    {
        cout<<"Compartment Volume: "<<value<<endl;
    }
    else
    {
        cout<<getLastError()<<endl;
    }
    cout<<stringArrayToString(getCompartmentIds())<<endl;

    getRateOfChange(0, value);
    cout<<"Rate of change:"<<value<<endl;

	
    //cout<<stringArrayToString(getFloatingSpeciesInitialConditionIds())<<endl;


    cout<<" ---- getElasticityCoefficientNames ---\n"<<listToString(getElasticityCoefficientIds())<<endl;
//    cout<<stringArrayToString(getRateOfChangeIds())<<endl;
    setCapabilities (NULL);
    cout<<getCapabilities()<<endl;

//    C_DECL_SPEC bool                    rrCallConv   getScaledFloatingSpeciesElasticity(const char* reactionName, const char* speciesName, double& value);
    if(getScaledFloatingSpeciesElasticity("_J1", "S1", value))
    {
        cout<<"ScaledFloatingSpeciesElasticity "<<value<<endl;
    }
    else
    {
        cout<<getLastError()<<endl;
    }

    cout<<"getFloatingSpeciesInitialConditionNames: "<<stringArrayToString(getFloatingSpeciesInitialConditionIds())<<endl;


    cout<<getCurrentSBML();
	///////////////////
    text = getCopyright();
    if(hasError())
    {
        char* error = getLastError();
        cout<<error<<endl;
    }

    cout<<text<<endl;
    freeText(text);
    freeRRInstance(rrHandle);
    return 0;
}

void printMatrix(char* msg1, RRMatrixHandle mat)
{
	cout<<msg1<<"\n";
	cout<<("------------\n\n");
    char *text = matrixToString(mat);
    if(text)
    {
		cout<<text<<"\n\n";
        freeText(text);
    }
    else
    {
    	cout<<"NULL\n\n";
    }
}
