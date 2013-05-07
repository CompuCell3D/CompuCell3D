#pragma hdrstop
#include <algorithm>
#include "sbml/Model.h"
#include "sbml/SBMLDocument.h"
#include "rr-libstruct/lsLibStructural.h"
#include "rrStringListContainer.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrRule.h"
#include "rrScanner.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrCGenerator.h"
#include "rrCSharpGenerator.h"
//---------------------------------------------------------------------------

using namespace std;
using namespace ls;

namespace rr
{

CGenerator::CGenerator(LibStructural& ls, NOMSupport& nom)
:
ModelGenerator(ls, nom)
{}

CGenerator::~CGenerator(){}

int CGenerator::getNumberOfFloatingSpecies()
{
    return mFloatingSpeciesConcentrationList.size();    //Todo: is there a list of floating species?
}

string CGenerator::getHeaderCode()
{
    return mHeader.ToString();
}

string CGenerator::getSourceCode()
{
    return mSource.ToString();
}

string CGenerator::getHeaderCodeFileName()
{
    return mHeaderCodeFileName;
}

string CGenerator::getSourceCodeFileName()
{
    return mSourceCodeFileName;
}


// Generates the Model Code from the SBML string
string CGenerator::generateModelCode(const string& sbmlStr, const bool& _computeAndAssignConsevationLaws)
{
	//This function now assume that the sbml already been loaded into NOM and libstruct..
	mComputeAndAssignConsevationLaws  = _computeAndAssignConsevationLaws;
    Log(lDebug2)<<"Entering CGenerators generateModelCode function";
    StringList  Warnings;
    CodeBuilder ignore;     //The Write functions below are inherited with a CodeBuilder in the
                            //prototype that is not to be used..

    //Clear header and source file objects...
    mHeader.Clear();
    mSource.Clear();

    mModelName = mNOM.getModelName();
    if(!mModelName.size())
    {
        Log(lWarning)<<"Model name is empty. ModelName is assigned 'NameNotSet'.";
        mModelName = "NameNotSet";
    }

    Log(lDebug1)<<"Processing model: "<<mModelName;
    mNumReactions  = mNOM.getNumReactions();

    Log(lDebug3)<<"Number of reactions:"<<mNumReactions;

    mGlobalParameterList.Clear();
    mModifiableSpeciesReferenceList.Clear();
    mLocalParameterList.reserve(mNumReactions);
    mReactionList.Clear();
    mBoundarySpeciesList.Clear();
    mFloatingSpeciesConcentrationList.Clear();
    mFloatingSpeciesAmountsList.Clear();
    mCompartmentList.Clear();
    mConservationList.Clear();
    mFunctionNames.empty();
    mFunctionParameters.empty();

	if(mComputeAndAssignConsevationLaws)
    {
        mNumIndependentSpecies 	= mLibStruct.getNumIndSpecies();
        mIndependentSpeciesList = mLibStruct.getIndependentSpecies();
        mDependentSpeciesList   = mLibStruct.getDependentSpecies();
    }
    else
    {
        mNumIndependentSpecies = mLibStruct.getNumSpecies();
        mIndependentSpeciesList = mLibStruct.getSpecies();
    }

    // Load the compartment array (name and value)
    mNumCompartments            = readCompartments();

    // Read FloatingSpecies
    mNumFloatingSpecies         = readFloatingSpecies();
    mNumDependentSpecies        = mNumFloatingSpecies - mNumIndependentSpecies;

    // Load the boundary species array (name and value)
    mNumBoundarySpecies     = readBoundarySpecies();

    // Get all the parameters into a list, global and local
    mNumGlobalParameters     = readGlobalParameters();
    mNumModifiableSpeciesReferences = readModifiableSpeciesReferences();

    // Load up local parameters next
    readLocalParameters(mNumReactions, mLocalParameterDimensions, mTotalLocalParmeters);
    mNumEvents = mNOM.getNumEvents();

    //Write model to String builder...
    writeClassHeader(ignore);
    writeOutVariables(ignore);
    writeOutSymbolTables(ignore);

    ///// Write non exports
    mHeader.NewLine("\n//NON - EXPORTS ========================================");
    mHeader.AddFunctionProto("void", "InitializeDelays(ModelData* md)");

    ///// Start of exported functions
    mHeader.NewLine("\n//EXPORTS ========================================");
    mHeader.AddFunctionExport("int", "InitModelData(ModelData* md)");
    mHeader.AddFunctionExport("int", "InitModel(ModelData* md)");

    mHeader.AddFunctionExport("char*", "getModelName(ModelData* md)");
    ///////////////

    writeResetEvents(ignore, mNumEvents);
    writeSetConcentration(ignore);
    writeGetConcentration(ignore);
    writeConvertToAmounts(ignore);
    writeConvertToConcentrations(ignore);
    writeProperties(ignore);
    writeAccessors(ignore);
    writeUserDefinedFunctions(ignore);
    writeSetInitialConditions(ignore, mNumFloatingSpecies);
    writeSetBoundaryConditions(ignore);
    writeSetCompartmentVolumes(ignore);
    writeSetParameterValues(ignore, mNumReactions);
    writeComputeConservedTotals(ignore, mNumFloatingSpecies, mNumDependentSpecies);

    // Get the L0 matrix
    int nrRows;
    int nrCols;

    ls::DoubleMatrix* aL0 = initializeL0(nrRows, nrCols);     //Todo: What is this doing? answer.. it is used below..
    writeUpdateDependentSpecies(ignore, mNumIndependentSpecies, mNumDependentSpecies, *aL0);
    int numOfRules = writeComputeRules(ignore, mNumReactions);

    writeComputeAllRatesOfChange(ignore, mNumIndependentSpecies, mNumDependentSpecies, *aL0);
	delete aL0;
    writeComputeReactionRates(ignore, mNumReactions);
    writeEvalModel(ignore, mNumReactions, mNumIndependentSpecies, mNumFloatingSpecies, numOfRules);
    writeEvalEvents(ignore, mNumEvents, mNumFloatingSpecies);
    writeEventAssignments(ignore, mNumReactions, mNumEvents);
    writeEvalInitialAssignments(ignore, mNumReactions);
    writeTestConstraints(ignore);

    writeInitModelDataFunction(mHeader, mSource);
    writeInitFunction(mHeader, mSource);

    mHeader<<"\n\n#endif //modelH"<<NL();
	string modelCode = mHeader.ToString() + mSource.ToString();

    Log(lDebug5)<<" ------ Model Code --------\n"
            <<modelCode
            <<" ----- End of Model Code -----\n";

    return modelCode;
}

void CGenerator::writeClassHeader(CodeBuilder& ignore)
{
    //Create c code header file....
    mHeader<<"#ifndef modelH"<<endl;
    mHeader<<"#define modelH"<<endl;
    mHeader<<"#include <stdio.h>"<<endl;
    mHeader<<"#include <stdbool.h>"<<endl;
    mHeader<<"#include \"rrModelData.h\"\t             //Contains the structure defining model data passed to the shared library."<<endl;
    mHeader<<"#include \"rrCExporter.h\"\t             //Export Stuff."<<endl;


    mHeader<<Append("//************************************************************************** " + NL());
    mHeader<<"//Number of floating species: "<<mFloatingSpeciesConcentrationList.size()<<endl;
    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        mHeader<<"\t// y["<<i<<"] = "<<mFloatingSpeciesConcentrationList[i].name<<endl;//{2}", NL());
    }

    mHeader<<Append("//************************************************************************** " + NL());
    mHeader<<Append(NL());
//    mHeader<<Format("D_S struct TModel{0}", NL());
//    mHeader<<Append("{" + NL());

    //Header of the source file...
    mSource<<"#include <math.h>"<<endl;
    mSource<<"#include <stdio.h>"<<endl;
    mSource<<"#include <stdlib.h>"<<endl;
    mSource<<"#include <string.h>"<<endl;
    mSource<<"#include \"rrSupport.h\"\t     //Supportfunctions for event handling.."<<endl;
}

void CGenerator::writeOutVariables(CodeBuilder& ignore)
{}


void CGenerator::writeComputeAllRatesOfChange(CodeBuilder& ignore, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0)
{
     //In header
       mHeader.AddFunctionExport("void", "computeAllRatesOfChange(ModelData* md)");
    mSource<<Append("//Uses the equation: dSd/dt = L0 dSi/dt" + NL());
    mSource<<"void computeAllRatesOfChange(ModelData* md)\n{";

    mSource<<gNL<<gTab<<"int i;\n";
    mSource<<"\n\tdouble* dTemp = (double*) malloc( sizeof(double)* (md->amountsSize + md->rateRulesSize) );\n"; //Todo: Check this

    for (int i = 0; i < numAdditionalRates(); i++)
    {
        mSource<<Format("\tdTemp[{0}] = {1};{2}", i, mMapRateRule[i], NL());
    }

    mSource<<gTab<<"for(i = 0; i < md->amountsSize; i++)\n";
    mSource<<gTab<<"{\n"<<gTab<<gTab<<"dTemp[i + md->rateRulesSize] = md->amounts[i];\n\t}";
    mSource<<Append("\n\t//amounts.CopyTo(dTemp, rateRules.Length); " + NL());

    mSource<<Append("\t__evalModel(md, md->time, dTemp);" + NL());
    bool isThereAnEntry = false;
    for (int i = 0; i < numDependentSpecies; i++)
    {
        mSource<<Format("\tmd->dydt[{0}] = ", (numIndependentSpecies + i));
        isThereAnEntry = false;
        for (int j = 0; j < numIndependentSpecies; j++)
        {
            string dyName = Format("md->dydt[{0}]", j);

            if (L0(i,j) > 0)
            {
                isThereAnEntry = true;
                if (L0(i,j) == 1)
                {
                    mSource<<Format(" + {0};{1}", dyName, NL());
                }
                else
                {
                    mSource<<Format(" + (double){0}{1}{2};{3}", writeDouble(L0(i,j)), mFixAmountCompartments, dyName, NL());
                }
            }
            else if (L0(i,j) < 0)
            {
                isThereAnEntry = true;
                if (L0(i,j) == -1)
                {
                    mSource<<Format(" - {0};{1}", dyName, NL());
                }
                else
                {
                    mSource<<Format(" - (double){0}{1}{2};{3}", writeDouble(fabs(L0(i,j))), mFixAmountCompartments, dyName, NL());
                }
            }
        }
        if (!isThereAnEntry)
        {
            mSource<<Append("0;");
        }
        mSource<<"\n";
    }

    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeComputeConservedTotals(CodeBuilder& ignore, const int& numFloatingSpecies, const int& numDependentSpecies)
{
    mHeader.AddFunctionExport("void", "computeConservedTotals(ModelData* md)");
    mSource<<"// Uses the equation: C = Sd - L0*Si"<<endl;
    mSource<<"void computeConservedTotals(ModelData* md)\n{";

    if (numDependentSpecies > 0)
    {
        string factor;
        ls::DoubleMatrix *gamma = mLibStruct.getGammaMatrix();


        for (int i = 0; i < numDependentSpecies; i++)
        {
            mSource<<Format("\n\tmd->ct[{0}] = ", i);
            for (int j = 0; j < numFloatingSpecies; j++)
            {
                double current = (gamma != NULL) ? (*gamma)(i,j) : 1.0;    //Todo: This is a bug? We should not be here if the matrix is NULL.. Triggered by model 00029

                if ( current != 0.0 )
                {
                    if (!gamma)//IsNaN(current)) //C# code is doing one of these.. factor = "" .. ??
                    {
                        // TODO: fix this
                        factor = "";
                    }
                    else if (fabsl(current) == 1.0)
                    {
                        factor = "";
                    }
                    else
                    {
                        factor = writeDouble(fabsl(current)) +
                                 mFixAmountCompartments;
                    }

                    if (current > 0)
                    {
                        string cYY = convertSpeciesToY(mFloatingSpeciesConcentrationList[j].name);
                        string cTC = convertCompartmentToC(mFloatingSpeciesConcentrationList[j].compartmentName);
                        mSource<<Append(" + " + factor + "md->" + cYY +
                                  mFixAmountCompartments +
                                  convertCompartmentToC(mFloatingSpeciesConcentrationList[j].compartmentName));
                    }
                    else
                    {
                        mSource<<Append(" - " + factor + convertSpeciesToY(mFloatingSpeciesConcentrationList[j].name) +
                                  mFixAmountCompartments +
                                  convertCompartmentToC(mFloatingSpeciesConcentrationList[j].compartmentName));
                    }
                }
            }
            mSource<<Append(";" + NL());
            mConservationList.Add(Symbol("CSUM" + ToString(i))); //TODO: how to deal with this?
        }
    }
    else
    {
        //mSource<<"printf(\"In an empty ComputeConservedTotals!\\n \");\n";
    }
    mSource<<"}\n\n";
}

void CGenerator::writeUpdateDependentSpecies(CodeBuilder& ignore, const int& numIndependentSpecies, const int& numDependentSpecies, DoubleMatrix& L0)
{
    mHeader.AddFunctionExport("void", "updateDependentSpeciesValues(ModelData* md, double* y)");
    mSource<<Append("// Compute values of dependent species " + NL());
    mSource<<Append("// Uses the equation: Sd = C + L0*Si" + NL());
    mSource<<"void updateDependentSpeciesValues(ModelData* md, double* y)\n{";

    // Use the equation: Sd = C + L0*Si to compute dependent concentrations

    if (numDependentSpecies > 0)
    {
        for (int i = 0; i < numDependentSpecies; i++)
        {
            mSource<<Format("\n\tmd->y[{0}] = ", (i + numIndependentSpecies));
            mSource<<Format("(md->ct[{0}]", i);
            string cLeftName =
                convertCompartmentToC(
                    mFloatingSpeciesConcentrationList[i + numIndependentSpecies].compartmentName);

            for (int j = 0; j < numIndependentSpecies; j++)
            {
                string yName = Format("y[{0}]", j);
                string cName = convertCompartmentToC(mFloatingSpeciesConcentrationList[j].compartmentName);
                double* mat = L0.GetPointer();
                double matElementValue = L0(i,j);

                if (L0(i,j) > 0) // In C# code there is no checking for index out of bound..
                {
                    if (L0(i,j) == 1)
                    {
                        mSource<<Format(" + {0}\t{1}{2}{3}{0}\t",
                            "",
                            yName,
                            mFixAmountCompartments,
                            cName);
                    }
                    else
                    {
                        mSource<<Format("{0} + (double){1}{2}{3}{2}{4}",
                            "",
                            writeDouble(L0(i,j)),
                            mFixAmountCompartments,
                            yName,
                            cName);
                    }
                }
                else if (L0(i,j) < 0)
                {
                    if (L0(i,j) == -1)
                    {
                        mSource<<Format("{0} - {1}{2}{3}",
                            "",
                            yName,
                            mFixAmountCompartments,
                            cName);
                    }
                    else
                    {
                        mSource<<Format("{0} - (double){1}{2}{3}{2}{4}",
                            "",
                            writeDouble(fabsl(L0(i,j))),
                            mFixAmountCompartments,
                            yName,
                            cName);
                    }
                }
            }
            mSource<<Format(") / {0};{1}", cLeftName, NL());
        }
    }
    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeUserDefinedFunctions(CodeBuilder& ignore)
{
    for (int i = 0; i < mNOM.getNumFunctionDefinitions(); i++)
    {
        try
        {
            StringListContainer oList = mNOM.getNthFunctionDefinition(i);
            StringList aList = oList[0];

              string sName = aList[0];
              //sName.Trim();
            mFunctionNames.Add(sName);
            StringList oArguments = oList[1];
            StringList list2 = oList[2];
            string sBody = list2[0];

            mSource<<Format("// User defined function:  {0}{1}", sName, NL());
            mSource<<Format("\t double {0} (", sName);

            for (int j = 0; j < oArguments.Count(); j++)
            {
                mSource<<Append("double " + (string)oArguments[j]);
                mFunctionParameters.Add((string)oArguments[j]);
                if (j < oArguments.Count() - 1)
                {
                    mSource<<Append(", ");
                }
            }
            string userFunc = convertUserFunctionExpression(sBody);

            if(userFunc.find("spf_piecewise") != string::npos)
            {
                ConvertFunctionCallToUseVarArgsSyntax("spf_piecewise", userFunc);
            }

            if(userFunc.find("spf_and") != string::npos)
            {
                ConvertFunctionCallToUseVarArgsSyntax("spf_and", userFunc);
            }

            if(userFunc.find("spf_or") != string::npos)
            {
                ConvertFunctionCallToUseVarArgsSyntax("spf_or", userFunc);
            }

            if(userFunc.find("spf_xor") != string::npos)
            {
                ConvertFunctionCallToUseVarArgsSyntax("spf_xor", userFunc);
            }

            mSource<<Append(")" + NL() + "\t{" + NL() + "\t\t return " +
                      userFunc
                      + ";" + NL() + "\t}" + NL() + NL());
        }
        catch (const Exception& ex)
        {
            CodeBuilder msg;
            msg<<"Error while trying to get Function Definition #" << i <<ex.what() << "\r\n\r\n";
            throw Exception(msg.ToString());
        }
    }
}

void CGenerator::writeResetEvents(CodeBuilder& ignore, const int& numEvents)
{
      mHeader.AddFunctionExport("void", "resetEvents(ModelData* md)");
      mSource<<"void resetEvents(ModelData* md)\n{";
      for (int i = 0; i < numEvents; i++)
      {
          mSource<<Format("\n\tmd->eventStatusArray[{0}] = false;{1}", i, NL());
          mSource<<Format("\tmd->previousEventStatusArray[{0}] = false;", i);
          if(i == numEvents -1)
          {
              mSource<<"\n";
          }
      }
      mSource<<Format("}{0}", NL());
}

void CGenerator::writeSetConcentration(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "setConcentration(ModelData* md, int index, double value)");
    mSource<<"\nvoid setConcentration(ModelData* md, int index, double value)\n{";
    mSource<<Format("\n\tdouble volume = 0.0;{0}", NL());
    mSource<<Format("\tmd->y[index] = value;{0}", NL());
    mSource<<Format("\tswitch (index)\n\t{{0}", NL());

    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        mSource<<Format("\t\tcase {0}:\n\t\t\tvolume = {1};{2}",
          i,
          convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName),
          NL());
      mSource<<Format("\t\tbreak;{0}", NL());
    }

    mSource<<Format("\t}{0}", NL());

    mSource<<Format("\tmd->amounts[index] = md->y[index]*volume;{0}", NL());
    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeGetConcentration(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("double", "getConcentration(ModelData* md,int index)");
    mSource<<Format("double getConcentration(ModelData* md, int index)\n{{0}", NL());
    mSource<<Format("\treturn md->y[index];{0}", NL());
    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeConvertToAmounts(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "convertToAmounts(ModelData* md)");
    mSource<<Format("void convertToAmounts(ModelData* md)\n{{0}", NL());
    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        mSource<<Format("\tmd->amounts[{0}] = md->y[{0}]*{1};{2}",
            i,
            convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName),
            NL());
    }
    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeConvertToConcentrations(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "convertToConcentrations(ModelData* md)");
    mSource<<"void convertToConcentrations(ModelData* md)\n{";
    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        mSource<<"\n\tmd->y[" << i << "] = md->amounts[" << i << "] / " <<
                  convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName) << ";";
    }
    mSource<<Append("\n}" + NL() + NL());
}

void CGenerator::writeProperties(CodeBuilder& ignore)
{
}

void CGenerator::writeAccessors(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("int", "getNumLocalParameters(ModelData* md, int reactionId)");
    mSource<<"int getNumLocalParameters(ModelData* md, int reactionId)\n{\n\t";
    mSource<<"return md->localParameterDimensions[reactionId];\n}\n\n";
}

string CGenerator::findSymbol(const string& varName)
{
      int index = 0;
      if (mFloatingSpeciesConcentrationList.find(varName, index))
      {
          return Format("md->y[{0}]", index);
      }
      else if (mGlobalParameterList.find(varName, index))
      {
          return Format("md->gp[{0}]", index);
      }
      else if (mBoundarySpeciesList.find(varName, index))
      {
          return Format("md->bc[{0}]", index);
      }
      else if (mCompartmentList.find(varName, index))
      {
          return Format("md->c[{0}]", index);
      }
      else if (mModifiableSpeciesReferenceList.find(varName, index))
      {
          return Format("md->sr[{0}]", index);
      }
      else
      {
      	throw Exception(Format("Unable to locate lefthand side symbol in assignment[{0}]", varName));
      }
}

void CGenerator::writeTestConstraints(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "testConstraints(ModelData* md)");
    mSource<<Append("void testConstraints(ModelData* md)" + NL());
    mSource<<Append("{");

    for (int i = 0; i < mNOM.getNumConstraints(); i++)
    {
        string sMessage;
        string sCheck = mNOM.getNthConstraint(i, sMessage);

        mSource<<Append("\tif (" + substituteTerms(mNOM.getNumReactions(), "", sCheck) + " == 0.0 )" + NL());
        mSource<<Append("\t\tthrow new Exception(\"" + sMessage + "\");" + NL());
    }

    mSource<<Append("}" + NL() + NL());
}

void CGenerator::writeEvalInitialAssignments(CodeBuilder& ignore, const int& numReactions)
{
    mHeader.AddFunctionExport("void", "evalInitialAssignments(ModelData* md)");
    mSource<<Append("void evalInitialAssignments(ModelData* md)" + NL());
    mSource<<Append("{\n");

    int numInitialAssignments = mNOM.getNumInitialAssignments();

    if (numInitialAssignments > 0)
    {
        vector< pair<string, string> > oList;// = new List<Pair<string, string>>();
        for (int i = 0; i < numInitialAssignments; i++)
        {
            pair<string, string> pair = mNOM.getNthInitialAssignmentPair(i);
            oList.push_back(mNOM.getNthInitialAssignmentPair(i));
        }

        // sort them ...
        bool bChange = true;
        int nIndex = -1;
        while (bChange)
        {
            bChange = false;

            for (int i = 0; i < oList.size(); i++)
            {
                pair<string, string> current = oList[i];
                for (int j = i + 1; j < oList.size(); j++)
                {
                    if (expressionContainsSymbol(current.second, oList[j].first))
                    {
                        bChange = true;
                        nIndex = j;
                        break;
                    }
                }
                if (bChange)
                {
                    break;
                }
            }

            if (bChange)
            {
                pair<string, string> pairToMove = oList[nIndex];
                oList.erase(oList.begin() + nIndex);
                //oList.RemoveAt(nIndex);
                oList.insert(oList.begin(), pairToMove);    //Todo: check it this is correct...
            }
        }

        vector< pair<string, string> >::iterator iter;
        for(iter = oList.begin(); iter < oList.end(); iter++)
        {
            pair<string, string>& pair = (*iter);
            string leftSideRule = findSymbol(pair.first);
            string rightSideRule = pair.second;
            if (leftSideRule.size())
            {
                mSource<<Append(leftSideRule + " = ");
                string temp = Append(substituteTerms(numReactions, "", rightSideRule) + ";" + NL());
                mSource<<temp;
            }
        }
    }
    for (int i = 0; i < mNOM.GetModel()->getNumEvents(); i++)
    {
        libsbml::Event *current = mNOM.GetModel()->getEvent(i);
        string initialTriggerValue = ToString(current->getTrigger()->getInitialValue());//.ToString().ToLowerInvariant();
        mSource<<Append("\tmd->eventStatusArray[" + ToString(i) + "] = " + initialTriggerValue + ";" + NL());
        mSource<<Append("\tmd->previousEventStatusArray[" + ToString(i) + "] = " + initialTriggerValue + ";" + NL());
    }
    mSource<<Append("}" + NL() + NL());
}

int CGenerator::writeComputeRules(CodeBuilder& ignore, const int& numReactions)
{
    IntStringHashTable mapVariables;
    mapVariables.clear();
    int numRateRules = 0;
    int numOfRules = mNOM.getNumRules();

    mHeader.AddFunctionExport("void", "computeRules(ModelData* md, double* y)");
    mSource<<"void computeRules(ModelData* md, double* y)\n{\n";

    for (int i = 0; i < numOfRules; i++)
    {
        try
        {
            string leftSideRule = "";
            string rightSideRule = "";
            string ruleType = mNOM.getNthRuleType(i);

            // We only support assignment and ode rules at the moment
            string eqnRule = mNOM.getNthRule(i);
            RRRule aRule(eqnRule, ruleType);
            string varName       = Trim(aRule.GetLHS());
            string rightSide     = Trim(aRule.GetRHS());

            bool isRateRule = false;

            switch (aRule.GetType())
            {
                case rtAlgebraic:
                    Log(lWarning)<<"RoadRunner does not yet support algebraic rules in SBML, they will be ignored.";
                    leftSideRule = "";//NULL;
                break;

                case rtAssignment:
                    leftSideRule = findSymbol(varName);
                break;

                case rtRate:
                    isRateRule = true;
                    int index;
                    if (mFloatingSpeciesConcentrationList.find(varName,  index))
                    {
                        leftSideRule = Format("\n\tmd->dydt[{0}]", index);
                        mFloatingSpeciesConcentrationList[index].rateRule = true;
                    }
                    else
                    {
                        leftSideRule = "\n\tmd->rateRules[" + ToString(numRateRules) + "]";
                        mMapRateRule[numRateRules] = findSymbol(varName);
                        mapVariables[numRateRules] = varName;
                        numRateRules++;
                    }
                break;
            }

            // Run the equation through MathML to carry out any conversions (eg ^ to Pow)
            if(rightSide.size())
            {
                string rightSideMathml    = mNOM.convertStringToMathML(rightSide);
                rightSideRule             = mNOM.convertMathMLToString(rightSideMathml);
            }

            if (leftSideRule.size())
            {
                mSource<<gTab<<Append(leftSideRule + " = ");
                int speciesIndex;
                bool isSpecies = mFloatingSpeciesConcentrationList.find(varName, speciesIndex);

                Symbol* symbol = (speciesIndex != -1) ? &(mFloatingSpeciesConcentrationList[speciesIndex]) : NULL;
                string sCompartment;

                if(isRateRule && mNOM.MultiplyCompartment(varName, sCompartment) && (rightSide.find(sCompartment) == string::npos))
                {
                    string temp = Format("({0}) * {1};{2}", substituteTerms(numReactions, "", rightSideRule), findSymbol(sCompartment), NL());
                    //temp = ReplaceWord("time", "md->time", temp);
                    mSource<<temp;
                }
                else
                {
                    if (isSpecies && !isRateRule && symbol != NULL && symbol->hasOnlySubstance && symbol->compartmentName.size() != 0)
                    {
                        mSource<<Format("({0}) / {1};{2}", substituteTerms(numReactions, "", rightSideRule), findSymbol(symbol->compartmentName), NL());
                    }
                    else
                    {
                        string temp   = Format("{0};{1}", substituteTerms(numReactions, "", rightSideRule), NL());
                        //temp = ReplaceWord("time", "md->time", temp);

                        if(temp.find("spf_piecewise") != string::npos)
            			{
                			ConvertFunctionCallToUseVarArgsSyntax("spf_piecewise", temp);
            			}
                        temp = RemoveNewLines(temp);
                        mSource<<temp;
                    }
                }

                if (mNOM.IsCompartment(varName))
                {
                    mSource<<Append("\n\tconvertToConcentrations(md);\n");
                }
            }
        }
        catch (const Exception& e)
        {
            throw CoreException("Error while trying to get Rule #" + ToString(i) + e.Message());
        }
    }

    mSource<<Append("\n}" + NL() + NL());

//  mHeader.FormatArray("D_S double", "_rateRules", numRateRules, "Vector containing values of additional rate rules"); //Todo: why is t his here in nowhere?
//    mHeader<<"D_S int _rateRulesSize="<<numRateRules<<";           // Number of rateRules   \n"; //Todo: why is this here in nowhere?
    mHeader.AddFunctionExport("void", "InitializeRates(ModelData* md)");

    mSource<<"void InitializeRates(ModelData* md)\n{";

    for (int i = 0; i < numRateRules; i++)
    {
        mSource<<"\n\tmd->rateRules[" << i << "] = " << mMapRateRule[i] << ";" << NL();
    }

    mSource<<Append("}" + NL() + NL());

    mHeader.AddFunctionExport("void", "AssignRatesA(ModelData* md)");
    mSource<<Append("void AssignRatesA(ModelData* md)\n{" + NL());

    for (int i = 0; i < mMapRateRule.size(); i++)
    {
        if(!i)
        {
            mSource<<"\n";
        }
        mSource<<"\t"<<(string) mMapRateRule[i] << " = md->rateRules[" << i << "];\n";
    }

    mSource<<Append("}" + NL() + NL());

    mHeader.AddFunctionExport("void", "InitializeRateRuleSymbols(ModelData* md)");
    mSource<<"void InitializeRateRuleSymbols(ModelData* md) \n{";
    for (int i = 0; i < mMapRateRule.size(); i++)
    {
        if(!i)
        {
            mSource<<"\n";
        }

        string varName = (string) mapVariables[i];
        if(varName.size())
        {
        	double value = mNOM.getValue(varName);
	        if (!IsNaN(value))
    	    {
        	    mSource<<gTab<<mMapRateRule[i] << " = " << ToString(value, mDoubleFormat) << ";" << NL();
        	}
        }
    }

    mSource<<"}\n\n";
    mHeader.AddFunctionExport("void", "AssignRatesB(ModelData* md, double oRates[])");
    mSource<<"void AssignRatesB(ModelData* md, double oRates[])\n{";

    for (int i = 0; i < mMapRateRule.size(); i++)
    {
        if(!i)
        {
            mSource<<"\n";
        }

        mSource<< mMapRateRule[i] << " = oRates[" << i << "];" << NL();
    }

    mSource<<Append("}" + NL() + NL());
    mHeader.AddFunctionExport("double*", "GetCurrentValues(ModelData* md)");
    mSource<<"double* GetCurrentValues(ModelData* md)\n{";
    mSource<<"\n\tdouble* dResult = (double*) malloc(sizeof(double)*"<<numAdditionalRates()<<");\n";

    for (int i = 0; i < mMapRateRule.size(); i++)
    {
           if(!i)
        {
            mSource<<"\n";
        }

        mSource<<"\tdResult[" << i << "] = " << mMapRateRule[i] << ";" << NL();
    }
    mSource<<"\treturn dResult;\n";

    mSource<<Append("}" + NL() + NL());
    return numOfRules;
}

void CGenerator::writeComputeReactionRates(CodeBuilder& ignore, const int& numReactions)
{
    mHeader.AddFunctionExport("void", "computeReactionRates(ModelData* md, double time, double *y)");
    mSource<<Append("// Compute the reaction rates" + NL());
    mSource<<"void computeReactionRates(ModelData* md, double time, double *y)\n{";    //Todo: what is time doing here?


    for (int i = 0; i < numReactions; i++)
    {
        string kineticLaw = mNOM.getKineticLaw(i);

        // The following code is for the case when the kineticLaw contains a ^ in place
        // of pow for exponent handling. It would not be needed in the case when there is
        // no ^ in the kineticLaw.
        string subKineticLaw;
//        if (kineticLaw.IndexOf("^", System.StringComparison.Ordinal) > 0) //Todo: fix this...
//        {
//            string kineticLaw_mathml = mNOM.convertStringToMathML(kineticLaw);
//            subKineticLaw = mNOM.convertMathMLToString(kineticLaw_mathml);
//        }
//        else
        {
            subKineticLaw = kineticLaw;
        }

        string modKineticLaw = substituteTerms(mReactionList[i].name, subKineticLaw, true) + ";";

        // modify to use current y ...
        modKineticLaw = Substitute(modKineticLaw, "_y[", "y[");
        string expression = Format("\n\tmd->rates[{0}] = {1}{2}", i, modKineticLaw, NL());

        if(expression.find("spf_and") != string::npos)
        {
            ConvertFunctionCallToUseVarArgsSyntax("spf_and", expression);
        }

        if(expression.find("spf_or") != string::npos)
        {
            ConvertFunctionCallToUseVarArgsSyntax("spf_or", expression);
        }

        if(expression.find("spf_xor") != string::npos)
        {
            ConvertFunctionCallToUseVarArgsSyntax("spf_xor", expression);
        }

        if(expression.find("spf_squarewave") != string::npos)
        {
            ConvertFunctionCallToUseVarArgsSyntax("spf_squarewave", expression);
        }

		if(expression.find("spf_piecewise") != string::npos)
        {
            ConvertFunctionCallToUseVarArgsSyntax("spf_piecewise", expression);
        }

        expression = RemoveChars(expression, "\t \n");
        mSource<<"\n\t"<<expression<<"\n";
    }

    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeEvalEvents(CodeBuilder& ignore, const int& numEvents, const int& numFloatingSpecies)
{
    mSource<<Append("//Event handling function" + NL());
    mHeader.AddFunctionExport("void", "evalEvents(ModelData* md, double timeIn, double *oAmounts)");
    mSource<<Append("void evalEvents(ModelData* md, double timeIn, double *oAmounts)" + NL());
    mSource<<Append("{" + NL());

    if (numEvents > 0)
    {
        for (int i = 0; i < numAdditionalRates(); i++)
        {
            mSource<<gTab<<(string) mMapRateRule[i] << " = oAmounts[" << i << "];" << NL();
        }
        for (int i = 0; i < numFloatingSpecies; i++)
        {
            mSource<<"\tmd->y[" << i << "] = oAmounts[" << (i + numAdditionalRates()) << "]/" <<
                      convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName) << ";" << NL();
        }
    }

    mSource<<Append("\tmd->time = timeIn;" + NL());
    mSource<<Append("\tupdateDependentSpeciesValues(md, md->y);" + NL());
    mSource<<Append("\tcomputeRules(md, md->y);" + NL());

    for (int i = 0; i < numEvents; i++)
    {
        ArrayList ev = mNOM.getNthEvent(i);
        StringList tempList = ev[0];
        string eventString = tempList[0];

        eventString = substituteTerms(0, "", eventString);
        mSource<<"\tmd->previousEventStatusArray[" << i << "] = md->eventStatusArray[" << i << "];" << NL();
        ConvertFunctionCallToUseVarArgsSyntax("spf_and", eventString);
        eventString = RemoveNewLines(eventString);
        mSource<<Append("\tif (" + eventString + " == 1.0)\n\t{" + NL());
        mSource<<Append("\t\tmd->eventStatusArray[" + ToString(i) + "] = true;" + NL());
        mSource<<Append("\t\tmd->eventTests[" + ToString(i) + "] = 1;" + NL());
        mSource<<Append("\n\t}\n\telse\n\t{\n");
        mSource<<Append("\t\tmd->eventStatusArray[" + ToString(i) + "] = false;" + NL());
        mSource<<Append("\t\tmd->eventTests[" + ToString(i) + "] = -1;" + NL());
        mSource<<Append("\t}" + NL());
    }
    mSource<<Append("}" + NL() + NL());
}

void CGenerator::writeEvalModel(CodeBuilder& ignore, const int& numReactions, const int& numIndependentSpecies, const int& numFloatingSpecies, const int& numOfRules)
{
    mHeader.AddFunctionExport("void", "__evalModel(ModelData* md, double, double*)");
    mSource<<Append("//Model Function" + NL());
    mSource<<"void __evalModel(ModelData* md, double timein, double* oAmounts)\n{";


    for (int i = 0; i < numAdditionalRates(); i++)
    {
        mSource<<"\n"<<(string)mMapRateRule[i] << " = oAmounts[" << i << "];" << NL();
    }

    for (int i = 0; i < numFloatingSpecies; i++)
    {
        mSource<<"\n\tmd->y[" << i << "] = oAmounts[" << i + numAdditionalRates() << "]/" <<
                  convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName) << ";" << NL();
    }

    mSource<<Append(NL());
    mSource<<Append("\tconvertToAmounts(md);" + NL());
    mSource<<Append("\tmd->time = timein;  // Don't remove" + NL());
    mSource<<Append("\tupdateDependentSpeciesValues(md, md->y);" + NL());

    if (numOfRules > 0)
    {
        mSource<<Append("\tcomputeRules(md, md->y);" + NL());
    }

    mSource<<Append("\tcomputeReactionRates(md, md->time, md->y);" + NL());

    // write out the ODE equations
    string stoich;
    for (int i = 0; i < numIndependentSpecies; i++)
    {
        CodeBuilder eqnBuilder;// = new CodeBuilder(" ");
        string floatingSpeciesName = mIndependentSpeciesList[i];
        for (int j = 0; j < numReactions; j++)
        {
            Reaction *oReaction = mNOM.GetModel()->getReaction(j);
            int numProducts = (int) oReaction->getNumProducts();
            double productStoichiometry;
            for (int k1 = 0; k1 < numProducts; k1++)
            {
                SpeciesReference* product = oReaction->getProduct(k1);

                string productName = product->getSpecies();
                if (floatingSpeciesName == productName)
                {
                    productStoichiometry = product->getStoichiometry();

                    if (product->isSetId() && product->getLevel() > 2)
                    {
                        stoich = "(" +
                             substituteTerms(numReactions, "",
                                product->getId()) +
                             ") * ";
                    }
                    else if (product->isSetStoichiometry())
                    {
                        if (productStoichiometry != 1)
                        {
                            int denom = product->getDenominator();
                            if (denom != 1)
                            {
                                stoich = Format("((double){0}/(double){1})*", writeDouble(productStoichiometry), denom);
                            }
                            else
                            {
                                stoich = writeDouble(productStoichiometry) + '*';
                            }
                        }
                        else
                        {
                            stoich = "";
                        }
                    }
                    else
                    {
                        if (product->isSetStoichiometryMath() && product->getStoichiometryMath()->isSetMath())
                        {
                            stoich = "(" +
                                     substituteTerms(numReactions, "",
                                        SBML_formulaToString(product->getStoichiometryMath()->getMath())) +
                                     ") * ";
                        }
                        else
                        {
                            stoich = "";
                        }
                    }
                    eqnBuilder<<Format(" + {0}md->rates[{1}]", stoich, j);
                }
            }

            int numReactants = (int)oReaction->getNumReactants();
            double reactantStoichiometry;
            for (int k1 = 0; k1 < numReactants; k1++)
            {
                SpeciesReference *reactant = oReaction->getReactant(k1);
                string reactantName = reactant->getSpecies();
                if (floatingSpeciesName == reactantName)
                {
                    reactantStoichiometry = reactant->getStoichiometry();

                    if (reactant->isSetId() && reactant->getLevel() > 2)
                    {
                        stoich = Format("({0}) * ", substituteTerms(numReactions, "", reactant->getId()));
                    }
                    else if (reactant->isSetStoichiometry())
                    {
                        if (reactantStoichiometry != 1)
                        {
                            int denom = reactant->getDenominator();
                            if (denom != 1)
                            {
                                stoich = Format("((double){0}/(double){1})*", writeDouble(reactantStoichiometry), denom);
                            }
                            else
                            {
                                stoich = writeDouble(reactantStoichiometry) + "*";
                            }
                        }
                        else
                        {
                            stoich = "";
                        }
                    }

                    else
                    {
                        if (reactant->isSetStoichiometryMath() && reactant->getStoichiometryMath()->isSetMath())
                        {
                            stoich = "(" +
                                     substituteTerms(numReactions, "",
                                        SBML_formulaToString(reactant->getStoichiometryMath()->getMath())) +
                                     ") * ";
                        }
                        else
                        {
                            stoich = "";
                        }
                    }

                    eqnBuilder<<Append(Format(" - {0}md->rates[{1}]", stoich, j));
                }
            }
        }

        string finalStr = eqnBuilder.ToString();//.Trim();

        if (IsNullOrEmpty(finalStr))
        {
            finalStr = "    0.0";
        }

        if (mNOM.GetSBMLDocument()->getLevel() > 2)
        {
            // remember to take the conversion factor into account
            string factor = "";
            Species* species = mNOM.GetModel()->getSpecies(floatingSpeciesName);
            if (species != NULL)
            {
                if (species->isSetConversionFactor())
                {
                    factor = species->getConversionFactor();
                }
                else if (mNOM.GetModel()->isSetConversionFactor())
                {
                    factor = mNOM.GetModel()->getConversionFactor();
                }
            }

            if (!IsNullOrEmpty(factor))
            {
                finalStr = findSymbol(factor) + " * (" + finalStr + ")";
            }
        }

        // If the floating species has a raterule then prevent the dydt
        // in the model function from overriding it. I think this is expected behavior.
        if (!mFloatingSpeciesConcentrationList[i].rateRule)
        {
            mSource<<"\tmd->dydt[" << i << "] =" << finalStr << ";" << NL();
        }
    }

    mSource<<Append("\tconvertToAmounts(md);" + NL());
    mSource<<Append("}" + NL() + NL());
}

void CGenerator::writeEventAssignments(CodeBuilder& ignore, const int& numReactions, const int& numEvents)
{
    StringList delays;
    vector<bool> eventType;
    vector<bool> eventPersistentType;
    if (numEvents > 0)
    {
        //Get array of pointers functions
        mSource<<("TEventAssignmentDelegate* Get_eventAssignments(ModelData* md) \n{\n\treturn md->eventAssignments;\n}\n\n");
        mSource<<("TPerformEventAssignmentDelegate* Get_performEventAssignments(ModelData* md) \n{\n\treturn md->performEventAssignments;\n}\n\n");
        mSource<<("TComputeEventAssignmentDelegate* Get_computeEventAssignments(ModelData* md) \n{\n\treturn md->computeEventAssignments;\n}\n\n");
        mSource<<("TEventDelayDelegate* GetEventDelays(ModelData* md) \n{\n\treturn md->eventDelays;\n}\n\n");
        mSource<<Append("// Event assignments" + NL());
        for (int i = 0; i < numEvents; i++)
        {
            ArrayList ev = mNOM.getNthEvent(i);
            eventType.push_back(mNOM.getNthUseValuesFromTriggerTime(i));
            eventPersistentType.push_back(mNOM.GetModel()->getEvent(i)->getTrigger()->getPersistent());

            StringList event = ev[1];
            int numItems = event.Count();
            string str = substituteTerms(numReactions, "", event[0]);
            delays.Add(str);

            mSource<<Format("void eventAssignment_{0}(ModelData* md) \n{{1}", i, NL());

            string funcName(Format("performEventAssignment_{0}(ModelData* md, double* values)", i));
            mHeader.AddFunctionExport("void", funcName);
            mSource<<Format("\tperformEventAssignment_{0}(md, computeEventAssignment_{0}(md) );{1}", i, NL());
            mSource<<Append("}\n\n");

            funcName = (Format("computeEventAssignment_{0}(ModelData* md)", i));
            mHeader.AddFunctionExport("double*", funcName);

            mSource<<Format("double* computeEventAssignment_{0}(ModelData* md)\n{{1}", i, NL());
            StringList oTemp;
            StringList oValue;
            int nCount = 0;
            int numAssignments = ev.Count() - 2;

            mSource<<Format("\t\tdouble* values = (double*) malloc(sizeof(double)*{0});{1}", numAssignments, NL());
            for (int j = 2; j < ev.Count(); j++)
            {
                StringList asgn = (StringList) ev[j];
                //string assignmentVar = substituteTerms(numReactions, "", (string)asgn[0]);
                string assignmentVar = findSymbol((string)asgn[0]);
                string str;
                Symbol *species = getSpecies(assignmentVar);


                if (species != NULL && species->hasOnlySubstance)
                {
                    str = Format("{0} = ({1}) / {2}", assignmentVar, substituteTerms(numReactions, "", (string)asgn[1]), findSymbol(species->compartmentName));
                }
                else
                {
                    str = Format("{0} = {1}", assignmentVar, substituteTerms(numReactions, "", (string) asgn[1]));
                }

                string sTempVar = Format("values[{0}]", nCount);

                oTemp.Add(assignmentVar);
                oValue.Add(sTempVar);

                str = sTempVar+ str.substr(str.find(" ="));
                nCount++;
                string temp = Format("\t\t{0};{1}", str, NL());
                mSource<<temp;
            }
            mSource<<Append("\treturn values;" + NL());
            mSource<<Append("}" + NL());
            mSource<<Format("void performEventAssignment_{0}(ModelData* md, double* values) \n{{1}", i, NL());

            for (int j = 0; j < oTemp.Count(); j++)
            {
                mSource<<Format("\t\t{0} = values[{1}];{2}", oTemp[j], j, NL());
                string aStr = (string) oTemp[j];
                aStr = Trim(aStr);

                if (StartsWith(aStr, "md->c[")) //Todo:May have to trim?
                {
                    mSource<<Append("\t\tconvertToConcentrations(md);" + NL());
                }
            }

            mSource<<Append("}" + NL());
        }
        mSource<<Append("\t" + NL());
    }

    //Have to create TEventDelegate functions here
    for (int i = 0; i < delays.Count(); i++)
    {
        mSource<<"double GetEventDelay_"<<i<<"(ModelData* md)\n{\n\treturn "<<delays[i]<<";\n}\n\n";
    }

    mSource<<"void InitializeDelays(ModelData* md)\n{\n";

    for (int i = 0; i < delays.Count(); i++)
    {
        mSource<<Format("\tmd->eventDelays[{0}] = (TEventDelayDelegate) malloc(sizeof(TEventDelayDelegate) * 1);{2}", i, delays[i], NL());

        //Inititialize
        mSource<<Format("\tmd->eventDelays[{0}] = GetEventDelay_{0};\n", i);
        mSource<<Format("\tmd->eventType[{0}] = {1};{2}", i, ToString((eventType[i] ? true : false)), NL());
        mSource<<Format("\tmd->eventPersistentType[{0}] = {1};{2}", i, (eventPersistentType[i] ? "true" : "false"), NL());
    }
    mSource<<"}\n\n";

    mHeader.AddFunctionExport("void", "computeEventPriorities(ModelData* md)");
    mSource<<"void computeEventPriorities(ModelData* md)\n{";
    for (int i = 0; i < numEvents; i++)
    {
        libsbml::Event* current = mNOM.GetModel()->getEvent(i);

        if (current->isSetPriority() && current->getPriority()->isSetMath())
        {
            string priority = SBML_formulaToString(current->getPriority()->getMath());
            mSource<<"\n"<<Format("\tmd->eventPriorities[{0}] = {1};{2}", i, substituteTerms(numReactions, "", priority), NL());

        }
        else
        {
            mSource<<"\n"<<Format("\tmd->eventPriorities[{0}] = 0;{1}", i, NL());
        }
    }
    mSource<<Format("}{0}{0}", NL());
}

void CGenerator::writeSetParameterValues(CodeBuilder& ignore, const int& numReactions)
{
    mHeader.AddFunctionExport("void", "setParameterValues(ModelData* md)");
    mSource<<"void setParameterValues(ModelData* md)\n{";


    for (int i = 0; i < mGlobalParameterList.size(); i++)
    {
        //If !+INF
        string para = Format("\n\t{0} = (double){1};{2}",
                      convertSymbolToGP(mGlobalParameterList[i].name),
                      writeDouble(mGlobalParameterList[i].value),
                      NL());
        //If a parameter is INF, it means it is not initialized properly ??
        if(para.find("INF") == string::npos && para.find("NAN") == string::npos)
        {
            mSource<<para;
        }
    }

    // Initialize local parameter values
    for (int i = 0; i < numReactions; i++)
    {
        for (int j = 0; j < mLocalParameterList[i].size(); j++)
        {
            mSource<<Format("\n\t_lp[{0}][{1}] = (double){2};{3}", i, j, writeDouble(mLocalParameterList[i][j].value), NL());
        }
    }

    mSource<<Append("}" + NL() + NL());
}

void CGenerator::writeSetCompartmentVolumes(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "setCompartmentVolumes(ModelData* md)");
    mSource << "void setCompartmentVolumes(ModelData* md)\n{";

    for (int i = 0; i < mCompartmentList.size(); i++)
    {
        mSource<<Append("\n\t" + convertSymbolToC(mCompartmentList[i].name) + " = (double)" +
                  writeDouble(mCompartmentList[i].value) + ";" + NL());

        // at this point we also have to take care of all initial assignments for compartments as well as
        // the assignment rules on compartments ... otherwise we are in trouble :)
        stack<string> initializations = mNOM.GetMatchForSymbol(mCompartmentList[i].name);
        while (initializations.size() > 0)
        {
            string term(initializations.top());
            string sub = substituteTerms(mNumReactions, "", term);
            mSource<<Append("\t" + sub + ";" + NL());
            initializations.pop();
        }
    }

    mSource<<Append("}" + NL() + NL());
}

void CGenerator::writeSetBoundaryConditions(CodeBuilder& ignore)
{
    mHeader.AddFunctionExport("void", "setBoundaryConditions(ModelData* md)");
    mSource<<"void setBoundaryConditions(ModelData* md)\n{\n";

    for (int i = 0; i < mBoundarySpeciesList.size(); i++)
    {
        if (IsNullOrEmpty(mBoundarySpeciesList[i].formula))
        {
            mSource<<Append("\t" + convertSpeciesToBc(mBoundarySpeciesList[i].name) + " = (double)" +
                      writeDouble(mBoundarySpeciesList[i].value) + ";" + NL());
        }
        else
        {
            mSource<<Append("\t\t" + convertSpeciesToBc(mBoundarySpeciesList[i].name) + " = (double)" +
                      mBoundarySpeciesList[i].formula + ";" + NL());
        }
    }
    mSource<<Append("}" + NL() + NL());
}


void CGenerator::writeSetInitialConditions(CodeBuilder& ignore, const int& numFloatingSpecies)
{
    mHeader.AddFunctionExport("void", "initializeInitialConditions(ModelData* md)");
    mSource<<"void initializeInitialConditions(ModelData* md)\n{";

    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        if (IsNullOrEmpty(mFloatingSpeciesConcentrationList[i].formula))
        {
            mSource<<Append("\n\tmd->init_" + convertSpeciesToY(mFloatingSpeciesConcentrationList[i].name) + " = (double)" +
                      writeDouble(mFloatingSpeciesConcentrationList[i].value) + ";");
        }
        else
        {
            string formula = mFloatingSpeciesConcentrationList[i].formula;
            mSource<<Append("\n\tmd->init_" + convertSpeciesToY(mFloatingSpeciesConcentrationList[i].name) + " = (double) " +
                      formula + ";");
        }
    }

    mSource<<Append("\n}" + NL() + NL());

    // ------------------------------------------------------------------------------
    mHeader.AddFunctionExport("void", "setInitialConditions(ModelData* md)");
    mSource<<"void setInitialConditions(ModelData* md)";
    mSource<<"\n{";

    for (int i = 0; i < numFloatingSpecies; i++)
    {
        mSource<<"\n\tmd->y[" << i << "] =  md->init_y[" << i << "];";
        mSource<<"\n\tmd->amounts[" << i << "] = md->y[" << i << "]*" <<
                  convertCompartmentToC(mFloatingSpeciesConcentrationList[i].compartmentName) << ";" << NL();
    }
    mSource<<Append("}" + NL() + NL());
}

string CGenerator::convertSpeciesToY(const string& speciesName)
{
    int index;
    if (mFloatingSpeciesConcentrationList.find(speciesName, index))
    {
        return "y[" + ToString(index) + "]";
    }

    throw new CoreException("Internal Error: Unable to locate species: " + speciesName);
}

string CGenerator::convertSpeciesToBc(const string& speciesName)
{
    int index;
    if (mBoundarySpeciesList.find(speciesName, index))
    {
        return "md->bc[" + ToString(index) + "]";
    }

    throw CoreException("Internal Error: Unable to locate species: " + speciesName);
}

string CGenerator::convertCompartmentToC(const string& compartmentName)
{
    int index;
    if (mCompartmentList.find(compartmentName, index))
    {
        return "md->c[" + ToString(index) + "]";
    }

    throw CoreException("Internal Error: Unable to locate compartment: " + compartmentName);
}

string CGenerator::convertSymbolToGP(const string& parameterName)
{
    int index;
    if (mGlobalParameterList.find(parameterName, index))
    {
        return "md->gp[" + ToString(index) + "]";
    }
      throw CoreException("Internal Error: Unable to locate parameter: " + parameterName);
}

string CGenerator::convertSymbolToC(const string& compartmentName)
{
    int index;
    if (mCompartmentList.find(compartmentName, index))
    {
        return "md->c[" + ToString(index) + "]";
    }
      throw CoreException("Internal Error: Unable to locate compartment: " + compartmentName);
}

void CGenerator::writeOutSymbolTables(CodeBuilder& ignore)
{
    mSource<<Append("void loadSymbolTables(ModelData* md)\n{");

    int nrFuncs = 0;
    for (int i = 0; i < mFloatingSpeciesConcentrationList.size(); i++)
    {
        mSource<<Format("\n\tmd->variableTable[{0}] = \"{1}\";", i, mFloatingSpeciesConcentrationList[i].name);
        nrFuncs++;
    }

    for (int i = 0; i < mBoundarySpeciesList.size(); i++)
    {
        mSource<<Format("\n\tmd->boundaryTable[{0}] = \"{1}\";", i, mBoundarySpeciesList[i].name);
        nrFuncs++;
    }

    for (int i = 0; i < mGlobalParameterList.size(); i++)
    {
        string name = mGlobalParameterList[i].name;
           mSource<<Format("\n\tmd->globalParameterTable[{0}] = \"{1}\";", i, mGlobalParameterList[i].name);
        nrFuncs++;
    }
    if(nrFuncs > 0)
    {
        mSource<<"\n";
    }
    mSource<<Format("}{0}{0}", NL());
}

int CGenerator::readFloatingSpecies()
{
    // Load a reordered list into the variable list.
    StringList reOrderedList;

	if(mComputeAndAssignConsevationLaws)
    {
       reOrderedList = mLibStruct.getReorderedSpecies();
    }
    else
    {
        reOrderedList = mLibStruct.getSpecies();
    }

    StringListContainer oFloatingSpecies = mNOM.getListOfFloatingSpecies();

    for (int i = 0; i < reOrderedList.Count(); i++)
    {
        for (int j = 0; j < oFloatingSpecies.Count(); j++)
        {
            StringList oTempList = oFloatingSpecies[j];
              if(reOrderedList[i] != (const string&) oTempList[0])
              {
                  continue;
              }

            string compartmentName = mNOM.getNthFloatingSpeciesCompartmentName(j);
            bool bIsConcentration  = ToBool(oTempList[2]);
            double dValue = ToDouble(oTempList[1]);
            if (IsNaN(dValue))
            {
                  dValue = 0;
            }

            Symbol *symbol = NULL;
            if (bIsConcentration)
            {
              symbol = new Symbol(reOrderedList[i], dValue, compartmentName);
            }
            else
            {
              int nCompartmentIndex;
              mCompartmentList.find(compartmentName, nCompartmentIndex);

              double dVolume = mCompartmentList[nCompartmentIndex].value;
              if (IsNaN(dVolume))
              {
                dVolume = 1;
              }

              stringstream formula;
              formula<<ToString(dValue,mDoubleFormat)<<"/ md->c["<<nCompartmentIndex<<"]";

              symbol = new Symbol(reOrderedList[i],
                  dValue / dVolume,
                  compartmentName,
                  formula.str());
            }

            if(mNOM.GetModel())
            {
                Species *aSpecies = mNOM.GetModel()->getSpecies(reOrderedList[i]);
                if(aSpecies)
                {
                    symbol->hasOnlySubstance = aSpecies->getHasOnlySubstanceUnits();
                    symbol->constant = aSpecies->getConstant();
                }
            }
            else
            {
                //TODO: How to report error...?
                //Log an error...
                symbol->hasOnlySubstance = false;
            }
            Log(lDebug5)<<"Adding symbol to mFloatingSpeciesConcentrationList:"<<(*symbol);
            mFloatingSpeciesConcentrationList.Add(*(symbol));
            delete symbol;
            break;
          }
          //throw RRException("Reordered Species " + reOrderedList[i] + " not found.");
      }
      return oFloatingSpecies.Count();
}

int CGenerator::readBoundarySpecies()
{
    int numBoundarySpecies;
    StringListContainer oBoundarySpecies = mNOM.getListOfBoundarySpecies();
    numBoundarySpecies = oBoundarySpecies.Count(); // sp1.size();
    for (int i = 0; i < numBoundarySpecies; i++)
    {
        StringList oTempList     = oBoundarySpecies[i];
        string sName             = oTempList[0];
        string compartmentName     = mNOM.getNthBoundarySpeciesCompartmentName(i);
        bool bIsConcentration     = ToBool(oTempList[2]);
        double dValue             = ToDouble(oTempList[1]);
        if (IsNaN(dValue))
        {
            dValue = 0;
        }

        Symbol *symbol = NULL;
        if (bIsConcentration)
        {
            //Todo: memoryleak
            symbol = new Symbol(sName, dValue, compartmentName);
        }
        else
        {
            int nCompartmentIndex;
            double dVolume;
            if(mCompartmentList.find(compartmentName, nCompartmentIndex))
            {
                dVolume = mCompartmentList[nCompartmentIndex].value;
            }
            else
            {
                if (IsNaN(dVolume))
                {
                    dVolume = 1;
                }
            }
            stringstream formula;
            formula<<ToString(dValue, mDoubleFormat)<<"/ md->c["<<nCompartmentIndex<<"]";
            symbol = new Symbol(sName,
                                dValue / dVolume,
                                compartmentName,
                                formula.str());
        }

        if(mNOM.GetModel())
        {
            Species* species = mNOM.GetModel()->getSpecies(sName);
            if(species)
            {
                symbol->hasOnlySubstance = species->getHasOnlySubstanceUnits();
                symbol->constant = species->getConstant();
            }
        }
        else
        {
            //TODO: How to report error...?
            //Log an error...
            symbol->hasOnlySubstance = false;

        }
        mBoundarySpeciesList.Add(*symbol);
    }
    return numBoundarySpecies;
}

//This function is obsolete.. initialize all model data in roadrunner instead..
void CGenerator::writeInitModelDataFunction(CodeBuilder& ignore, CodeBuilder& source)
{
    source.Line("\n//Function to initialize the model data structure. Returns an integer indicating result");
    source.Line("int InitModelData(ModelData* md)");
    source.Line("{");
	source.Line("\tprintf(\"Size of md   %d\\n\",  (int) sizeof(md));");
//	source.Line("\tprintf(\"Size of SModelData  %d\",  (int) sizeof(SModelData));");
	source.Line("\tprintf(\"Size of ModelData   %d\\n\",  (int) sizeof(ModelData));");
//	source.Line("\tprintf(\"Size of ModelData*  %d\\n\", (int) sizeof(&ModelData));");
	source.Line("\tprintf(\"Size of ModelData.eventDelays  %d\\n\", (int) sizeof(md->eventDelays));");
    source.TLine("return 0;");
    source.Line("}");
    source.NewLine();
}

//This function is obsolete.. initialize all model data in roadrunner instead..
void CGenerator::writeInitFunction(CodeBuilder& ignore, CodeBuilder& source)
{
    source.Line("\n//Function to initialize the model data structure. Returns an integer indicating result");
    source.Line("int InitModel(ModelData* md)");
    source.Line("{");

//    source<<"\t"<<Append("InitModelData(md);" , NL());
    source<<"\t"<<Append("setCompartmentVolumes(md);" , NL());
    source<<"\t"<<Append("InitializeDelays(md);" , NL());

    // Declare any eventAssignment delegates
    if (mNumEvents > 0)
    {
        for (int i = 0; i < mNumEvents; i++)
        {
            string iStr = ToString(i);
            source<<Append("\tmd->eventAssignments[" + iStr + "] = eventAssignment_" + iStr +";" + NL());
            source<<Append("\tmd->computeEventAssignments[" + iStr + "] = (TComputeEventAssignmentDelegate) computeEventAssignment_" + iStr + ";" + NL());
            source<<Append("\tmd->performEventAssignments[" + iStr + "] = (TPerformEventAssignmentDelegate) performEventAssignment_" + iStr + ";" + NL());
        }

        source<<Append("\tresetEvents(md);" + NL());

        //Test to call a function
        source<<Append("\tmd->eventAssignments[0](md);\n");
        source<<Append(NL());
    }

    if (mNumModifiableSpeciesReferences > 0)
    {
        for (int i = 0; i < mModifiableSpeciesReferenceList.size(); i++)
        {
            source<<Append("\t\tmd->sr[" + ToString(i) + "] = " + writeDouble(mModifiableSpeciesReferenceList[i].value) + ";" + NL());
        }
        source<<Append(NL());
    }

    source.TLine("return 0;");
    source.Line("}");
    source.NewLine();
}

void CGenerator::write_getModelNameFunction(CodeBuilder& ignore, CodeBuilder& source)
{
    source.Line("char* getModelName(ModelData* md)");
    source<<"{"                                         <<endl;
    source.TLine("return md->modelName;");
    source<<"}"                                         <<endl;
    source.NewLine();
}

bool CGenerator::saveSourceCodeToFolder(const string& folder, const string& baseName)
{
    string fName 		= ExtractFileName(baseName);
    mHeaderCodeFileName = JoinPath(folder, fName);
    mHeaderCodeFileName = ChangeFileExtensionTo(mHeaderCodeFileName, ".h");

    ofstream outFile(mHeaderCodeFileName.c_str());
    if(!outFile)
    {
        throw(Exception("Failed to open file:" + mHeaderCodeFileName));
    }

    outFile<<getHeaderCode();
    Log(lDebug3)<<"Wrote header to file: "<<mHeaderCodeFileName;
    outFile.close();

    mSourceCodeFileName = ChangeFileExtensionTo(mHeaderCodeFileName, ".c");
    outFile.open(mSourceCodeFileName.c_str());

    //We don't know the name of the file until here..
    //Write an include statement to it..
    vector<string> fNameParts = SplitString(mSourceCodeFileName,"\\");
    string headerFName = fNameParts[fNameParts.size() - 1];

    headerFName = ChangeFileExtensionTo(headerFName, ".h");
    outFile<<"#include \""<<ExtractFileName(headerFName)<<"\"\n"<<endl;
    outFile<<getSourceCode();
    outFile.close();
    Log(lDebug3)<<"Wrote source code to file: "<<mSourceCodeFileName;

    return true;
}

string CGenerator::convertUserFunctionExpression(const string& equation)
{
    if(!equation.size())
    {
        Log(lError)<<"The equation string supplied to "<<__FUNCTION__<<" is empty";
        return "";
    }
    Scanner s;
    stringstream ss;
    ss<<equation;
    s.AssignStream(ss);
    s.startScanner();
    s.nextToken();
    CodeBuilder  mSource;

    try
    {
        while (s.token() != CodeTypes::tEndOfStreamToken)
           {
            string theToken = s.tokenString;
               switch (s.token())
               {
                case CodeTypes::tWordToken:
                    if(theToken == "pow")
                    {
                        mSource<<Append("spf_pow");
                    }
                    else if(theToken == "sqrt")
                    {
                        mSource<<Append("sqrt");
                      }
                    else if(theToken == "log")
                    {
                        mSource<<Append("spf_log");
                    }
                    else if(theToken == "log10")
                    {
                        mSource<<Append("Log10");
                    }
                    else if(theToken == "floor")
                    {
                        mSource<<Append("spf_floor");
                    }
                    else if(theToken == "ceil")
                    {
                        mSource<<Append("spf_ceil");
                    }
                    else if(theToken == "factorial")
                    {
                        mSource<<Append("spf_factorial");
                    }
                    else if(theToken == "exp")
                    {
                        mSource<<Append("Math.Exp");
                    }
                    else if(theToken == "sin")
                    {
                        mSource<<Append("sin");
                    }
                    else if(theToken == "cos")
                    {
                        mSource<<Append("cos");
                    }
                    else if(theToken == "tan")
                    {
                        mSource<<Append("tan");
                    }
                    else if(theToken == "abs")
                    {
                        mSource<<Append("spf_abs");
                    }
                    else if(theToken == "asin")
                    {
                        mSource<<Append("asin");
                    }
                    else if(theToken == "acos")
                    {
                        mSource<<Append("acos");
                    }
                    else if(theToken == "atan")
                    {
                        mSource<<Append("atan");
                    }
                    else if(theToken == "sec")
                    {
                        mSource<<Append("MathKGI.Sec");
                    }
                    else if(theToken == "csc")
                    {
                        mSource<<Append("MathKGI.Csc");
                    }
                    else if(theToken == "cot")
                    {
                        mSource<<Append("MathKGI.Cot");
                    }
                    else if(theToken == "arcsec")
                    {
                        mSource<<Append("MathKGI.Asec");
                    }
                    else if(theToken == "arccsc")
                    {
                        mSource<<Append("MathKGI.Acsc");
                    }
                    else if(theToken == "arccot")
                    {
                        mSource<<Append("MathKGI.Acot");
                    }
                    else if(theToken == "sinh")
                    {
                        mSource<<Append("Math.Sinh");
                    }
                    else if(theToken == "cosh")
                    {
                        mSource<<Append("Math.Cosh");
                    }
                    else if(theToken == "tanh")
                    {
                        mSource<<Append("Math.Tanh");
                    }
                    else if(theToken == "arcsinh")
                    {
                        mSource<<Append("MathKGI.Asinh");
                    }
                    else if(theToken == "arccosh")
                    {
                        mSource<<Append("MathKGI.Acosh");
                    }
                    else if(theToken == "arctanh")
                    {
                        mSource<<Append("MathKGI.Atanh");
                    }
                    else if(theToken == "sech")
                    {
                        mSource<<Append("MathKGI.Sech");
                    }
                    else if(theToken == "csch")
                    {
                        mSource<<Append("MathKGI.Csch");
                    }
                    else if(theToken == "coth")
                    {
                        mSource<<Append("MathKGI.Coth");
                    }
                    else if(theToken == "arcsech")
                    {
                        mSource<<Append("MathKGI.Asech");
                    }
                    else if(theToken == "arccsch")
                    {
                        mSource<<Append("MathKGI.Acsch");
                    }
                    else if(theToken == "arccoth")
                    {
                               mSource<<Append("MathKGI.Acoth");
                    }
                    else if(theToken == "pi")
                    {
                        mSource<<Append("PI");
                    }
                    else if(theToken == "exponentiale")
                    {
                        mSource<<Append("Math.E");
                    }
                    else if(theToken == "avogadro")
                    {
                        mSource<<Append("6.02214179e23");
                    }
                    else if(theToken == "true")
                    {
                        mSource<<Append("1.0");
                    }
                    else if(theToken == "false")
                    {
                        mSource<<Append("0.0");
                    }
                    else if(theToken == "gt")
                    {
                        mSource<<Append("spf_gt");
                    }
                    else if(theToken == "lt")
                    {
                        mSource<<Append("spf_lt");
                    }
                    else if(theToken == "eq")
                    {
                        mSource<<Append("spf_eq");
                    }
                    else if(theToken == "neq")
                    {
                        mSource<<Append("spf_neq");
                    }
                    else if(theToken == "geq")
                    {
                        mSource<<Append("spf_geq");
                    }
                    else if(theToken == "leq")
                    {
                        mSource<<Append("spf_leq");
                    }
                    else if(theToken == "and")
                    {
                        mSource<<Append("supportFunction._and");
                    }
                    else if(theToken == "or")
                    {
                        mSource<<Append("supportFunction._or");
                    }
                    else if(theToken == "not")
                    {
                        mSource<<Append("supportFunction._not");
                    }
                    else if(theToken == "xor")
                    {
                        mSource<<Append("supportFunction._xor");
                    }
                    else if(theToken == "root")
                    {
                        mSource<<Append("spf_root");
                    }
					else if (theToken == "squarewave")
					{
						mSource<<Append("spf_squarewave");
					}
                    else if(theToken == "piecewise")
                    {
                        mSource<<Append("spf_piecewise");
                    }
                    else if (!mFunctionParameters.Contains(s.tokenString))
                    {
                        throw Exception("Token '" + s.tokenString + "' not recognized.");
                    }
                    else
                    {
                        mSource<<Append(s.tokenString);
                    }

                break; //Word token

                   case CodeTypes::tDoubleToken:
                       mSource<<Append(writeDouble(s.tokenDouble));
                       break;
                   case CodeTypes::tIntToken:
                    mSource<<Append((int) s.tokenInteger);
                       break;
                   case CodeTypes::tPlusToken:
                   mSource<<Append("+");
                   break;
                   case CodeTypes::tMinusToken:
                   mSource<<Append("-");
                   break;
                   case CodeTypes::tDivToken:
                   mSource<<Append("/");
                   break;
                   case CodeTypes::tMultToken:
                   mSource<<Append(mFixAmountCompartments);
                   break;
                   case CodeTypes::tPowerToken:
                   mSource<<Append("^");
                   break;
                   case CodeTypes::tLParenToken:
                   mSource<<Append("(");
                   break;
                   case CodeTypes::tRParenToken:
                   mSource<<Append(")");
                   break;
                   case CodeTypes::tCommaToken:
                   mSource<<Append(",");
                   break;
                   case CodeTypes::tEqualsToken:
                   mSource<<Append(" = ");
                   break;
                   case CodeTypes::tTimeWord1:
                   mSource<<Append("time");
                   break;
                   case CodeTypes::tTimeWord2:
                   mSource<<Append("time");
                   break;
                   case CodeTypes::tTimeWord3:
                   mSource<<Append("time");
                   break;
                   case CodeTypes::tAndToken:
                   mSource<<Append("spf_and");
                   break;
                   case CodeTypes::tOrToken:
                   mSource<<Append("spf_or");
                   break;
                   case CodeTypes::tNotToken:
                   mSource<<Append("spf_not");
                   break;
                   case CodeTypes::tLessThanToken:
                   mSource<<Append("spf_lt");
                   break;
                   case CodeTypes::tLessThanOrEqualToken:
                   mSource<<Append("spf_leq");
                   break;
                   case CodeTypes::tMoreThanOrEqualToken:
                   mSource<<Append("spf_geq");
                   break;
                   case CodeTypes::tMoreThanToken:
                   mSource<<Append("spf_gt");
                   break;
                   case CodeTypes::tXorToken:
                   mSource<<Append("spf_xor");
                   break;
                   default:
                   stringstream msg;
                   msg<< "Unknown token in convertUserFunctionExpression: " << s.tokenToString(s.token()) <<
                           "Exception raised in Module:roadRunner, Method:convertUserFunctionExpression";
                   throw Exception(msg.str());
               }
               s.nextToken();
        }
    }
    catch (const Exception& e)
    {
       throw CoreException(e.Message());
    }
    return mSource.ToString();
}

void CGenerator::substituteEquation(const string& reactionName, Scanner& s, CodeBuilder& mSource)
{
    string theToken(s.tokenString);
    if(theToken == "pow")
    {
        mSource<<Append("spf_pow");
    }
    else if(theToken == "sqrt")
    {
        mSource<<Append("sqrt");
    }
    else if(theToken == "log")
    {
        mSource<<Append("spf_log");
    }
    else if(theToken == "floor")
    {
        mSource<<Append("spf_floor");
    }
    else if(theToken == "ceil")
    {
        mSource<<Append("spf_ceil");
    }
    else if(theToken == "factorial")
    {
        mSource<<Append("spf_factorial");
    }
    else if(theToken == "log10")
    {
        mSource<<Append("spf_log10");
    }
    else if(theToken == "exp")
    {
        mSource<<Append("spf_exp");
    }
    else if(theToken == "abs")
    {
        mSource<<Append("spf_abs");
    }
    else if(theToken == "sin")
    {
        mSource<<Append("spf_sin");
    }
    else if(theToken == "cos")
    {
        mSource<<Append("cos");
    }
    else if(theToken == "tan")
    {
        mSource<<Append("tan");
    }
    else if(theToken == "asin")
    {
        mSource<<Append("asin");
    }
    else if(theToken == "acos")
    {
        mSource<<Append("acos");
    }
    else if(theToken == "atan")
    {
        mSource<<Append("atan");
    }
    else if(theToken == "sec")
    {
        mSource<<Append("sec");
    }
    else if(theToken == "csc")
    {
        mSource<<Append("csc");
    }
    else if(theToken == "cot")
    {
        mSource<<Append("cot");
    }
    else if(theToken == "arcsec")
    {
        mSource<<Append("asec");
    }
    else if(theToken == "arccsc")
    {
        mSource<<Append("arccsc");
    }
    else if(theToken == "arccot")
    {
        mSource<<Append("arccot");
    }
    else if(theToken == "sinh")
    {
        mSource<<Append("sinh");
    }
    else if(theToken == "cosh")
    {
        mSource<<Append("cosh");
    }
    else if(theToken == "tanh")
    {
        mSource<<Append("tanh");
    }
    else if(theToken == "arcsinh")
    {
        mSource<<Append("arcsinh");
    }
    else if(theToken == "arccosh")
    {
        mSource<<Append("arccosh");
    }
    else if(theToken == "arctanh")
    {
        mSource<<Append("arctanh");
    }
    else if(theToken == "sech")
    {
        mSource<<Append("sech");
    }
    else if(theToken == "csch")
    {
        mSource<<Append("csch");
    }
    else if(theToken == "coth")
    {
        mSource<<Append("coth");
    }
    else if(theToken == "arcsech")
    {
        mSource<<Append("arcsech");
    }
    else if(theToken == "arccsch")
    {
        mSource<<Append("arccsch");
    }
    else if(theToken == "arccoth")
    {
        mSource<<Append("arccoth");
    }
    else if(theToken == "pi")
    {
        mSource<<Append("PI");
    }
    else if(theToken == "avogadro")
    {
        mSource<<Append("6.02214179e23");
    }
    else if(theToken == "exponentiale")
    {
        mSource<<Append("E");
    }
    else if(theToken == "true")
    {
        //mSource<<Append("true");
        mSource<<Append("1.0");
    }
    else if(theToken == "false")
    {
        //mSource<<Append("false");
        mSource<<Append("0.0");
    }
    else if(theToken == "NaN")
    {
        mSource<<Append("NaN");
    }
    else if(theToken == "INF")
    {
        mSource<<Append("INF");
    }
    else if(theToken == "geq")
    {
        mSource<<Append("spf_geq");
    }
    else if(theToken == "leq")
    {
        mSource<<Append("spf_leq");
    }
    else if(theToken == "gt")
    {
        mSource<<Append("spf_gt");
    }
    else if(theToken == "lt")
    {
        mSource<<Append("spf_lt");
    }
    else if(theToken == "eq")
    {
        mSource<<Append("spf_eq");
    }
    else if(theToken == "neq")
    {
        mSource<<Append("spf_neq");
    }
    else if(theToken == "and")
    {
        mSource<<Append("spf_and");
    }
    else if(theToken == "or")
    {
        mSource<<Append("spf_or");
    }
    else if(theToken == "not")
    {
        mSource<<Append("spf_not");
    }
    else if(theToken == "xor")
    {
        mSource<<Append("spf_xor");
    }
    else if(theToken == "root")
    {
        mSource<<Append("spf_root");
    }
	else if(theToken == "squarewave")
	{
		mSource<<Append("spf_squarewave");
	}
    else if(theToken == "piecewise")
    {
        mSource<<Append("spf_piecewise");
    }
    else if(theToken == "delay")
    {
        mSource<<Append("spf_delay");
        Log(lWarning)<<"RoadRunner does not yet support delay differential equations in SBML, they will be ignored (i.e. treated as delay = 0).";
    }
    else
    {
        bool bReplaced = false;
        int index;
        if (mReactionList.find(reactionName, index))
        {
            int nParamIndex = 0;
            if (mLocalParameterList[index].find(s.tokenString, nParamIndex))
            {
                mSource<<Append("_lp[" + ToString(index) + "][" + ToString(nParamIndex) + "]");
                bReplaced = true;
            }
        }

        if (mBoundarySpeciesList.find(s.tokenString, index))
        {
            mSource<<Append("_bc[" + ToString(index) + "]");
            bReplaced = true;
        }
        if (!bReplaced &&
            (mFunctionParameters.Count() != 0 && !mFunctionParameters.Contains(s.tokenString)))
        {
            throw Exception("Token '" + s.tokenString + "' not recognized.");
        }
    }
}

void CGenerator::substituteWords(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& mSource)
{
    // Global parameters have priority
    int index;
    if (mGlobalParameterList.find(s.tokenString, index))
    {
        mSource<<Format("md->gp[{0}]", index);
    }
    else if (mBoundarySpeciesList.find(s.tokenString, index))
    {
        mSource<<Format("md->bc[{0}]", index);

        Symbol symbol = mBoundarySpeciesList[index];
        if (symbol.hasOnlySubstance)
        {
            // we only store concentration for the boundary so we better
            // fix that.
            int nCompIndex = 0;
            if (mCompartmentList.find(symbol.compartmentName, nCompIndex))
            {
                mSource<<Format("{0}_c[{1}]", mFixAmountCompartments, nCompIndex);
            }
        }
    }
    else if (mFloatingSpeciesConcentrationList.find(s.tokenString, index))
    {
        Symbol floating1 = mFloatingSpeciesConcentrationList[index];
        if (floating1.hasOnlySubstance)
        {
            mSource<<Format("md->amounts[{0}]", index);
        }
        else
        {
            mSource<<Format("md->y[{0}]", index);
        }
    }
    else if (mCompartmentList.find(s.tokenString, index))
    {
        mSource<<Format("md->c[{0}]", index);
    }
    else if (mFunctionNames.Contains(s.tokenString))
    {
        mSource<<Format("{0} ", s.tokenString);
    }
    else if (mModifiableSpeciesReferenceList.find(s.tokenString, index))
    {
        mSource<<Format("md->sr[{0}]", index);
    }
    else if (mReactionList.find(s.tokenString, index))
    {
        mSource<<Format("md->rates[{0}]", index);
    }
    else
    {
        substituteEquation(reactionName, s, mSource);
    }
}

void CGenerator::substituteToken(const string& reactionName, bool bFixAmounts, Scanner& s, CodeBuilder& mSource)
{
    string aToken = s.tokenString;
    CodeTypes codeType = s.token();
    switch(codeType)
    {
        case CodeTypes::tWordToken:
        case CodeTypes::tExternalToken:
        case CodeTypes::tExtToken:
            substituteWords(reactionName, bFixAmounts, s, mSource);
            break;

        case CodeTypes::tDoubleToken:
            mSource<<Append("(double) " + writeDouble(s.tokenDouble));
            break;
        case CodeTypes::tIntToken:
            mSource<<Append("(double)" + writeDouble((double)s.tokenInteger));
            break;
        case CodeTypes::tPlusToken:
            mSource<<Format("+{0}\t", NL());
            break;
        case CodeTypes::tMinusToken:
            mSource<<Format("-{0}\t", NL());
            break;
        case CodeTypes::tDivToken:
            mSource<<Format("/{0}\t", NL());
            break;
        case CodeTypes::tMultToken:
            mSource<<Format("*{0}\t", NL());
            break;
        case CodeTypes::tPowerToken:
            mSource<<Format("^{0}\t", NL());
            break;
        case CodeTypes::tLParenToken:
            mSource<<Append("(");
            break;
        case CodeTypes::tRParenToken:
            mSource<<Format("){0}\t", NL());
            break;
        case CodeTypes::tCommaToken:
            mSource<<Append(",");
            break;
        case CodeTypes::tEqualsToken:
            mSource<<Format(" = {0}\t", NL());
            break;
      case CodeTypes::tTimeWord1:
            mSource<<Append("md->time");
            break;
        case CodeTypes::tTimeWord2:
            mSource<<Append("md->time");
            break;
        case CodeTypes::tTimeWord3:
            mSource<<Append("md->time");
            break;
        case CodeTypes::tAndToken:
            mSource<<Format("{0}spf_and", NL());
            break;
        case CodeTypes::tOrToken:
            mSource<<Format("{0}spf_or", NL());
            break;
        case CodeTypes::tNotToken:
            mSource<<Format("{0}spf_not", NL());
            break;
        case CodeTypes::tLessThanToken:
            mSource<<Format("{0}spf_lt", NL());
            break;
        case CodeTypes::tLessThanOrEqualToken:
            mSource<<Format("{0}spf_leq", NL());
            break;
        case CodeTypes::tMoreThanOrEqualToken:
            mSource<<Format("{0}spf_geq", NL());
            break;
        case CodeTypes::tMoreThanToken:
            mSource<<Format("{0}spf_gt", NL());
            break;
        case CodeTypes::tXorToken:
            mSource<<Format("{0}spf_xor", NL());
            break;
        default:
        string aToken = s.tokenToString(s.token());
        Exception ae = Exception(
                 Format("Unknown token in substituteTerms: {0}", aToken,
                 "Exception raised in Module:roadRunner, Method:substituteTerms"));
         throw ae;
    }
}

}//Namespace

