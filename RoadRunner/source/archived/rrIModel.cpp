#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrIModel.h"
#include "rrLogger.h"
//---------------------------------------------------------------------------

namespace rr
{

IModel::IModel()
:
mDummyInt(0),
numIndependentVariables(&mDummyInt),
numDependentVariables(&mDummyInt),
numTotalVariables(&mDummyInt),
numBoundaryVariables(&mDummyInt),
numGlobalParameters(&mDummyInt),
numCompartments(&mDummyInt),
numReactions(&mDummyInt),
numRules(&mDummyInt),
numEvents(&mDummyInt),
time(0),
mModelName("NoNameSet")
{}

IModel::~IModel(){}


int IModel::getNumIndependentVariables()
{
    return *numIndependentVariables;
}

int IModel::getNumDependentVariables()
{
    return *numDependentVariables;
}

int IModel::getNumTotalVariables()
{
    return *numTotalVariables;
}

int IModel::getNumBoundarySpecies()
{
    return *numBoundaryVariables;    //Todos: bad naming - is Variables/Species, choose one..
}

int IModel::getNumGlobalParameters()
{
    return *numGlobalParameters;
}

int IModel::getNumCompartments()
{
    return *numCompartments;
}

int IModel::getNumReactions()
{
    return *numReactions;
}

int IModel::getNumRules()
{
    return *numRules;
}

int IModel::getNumEvents()
{
    return *numEvents;
}

//Virtual functions that should(?) be implemented in decendant..
//void  IModel::initializeInitialConditions(){}
void  IModel::setInitialConditions()                				{Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::setParameterValues()                                  {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::setBoundaryConditions()                               {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::InitializeRates()                                     {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::AssignRates()                                         {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::AssignRates(vector<double>& rates)                    {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::computeConservedTotals()                              {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::computeEventPriorites()                               {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::setConcentration(int index, double value)             {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::convertToAmounts()                                    {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
//void  IModel::convertToConcentrations() = 0;
void  IModel::updateDependentSpeciesValues(vector<double>& _y)      {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::computeRules(vector<double>& _y)                      {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::computeReactionRates(double time, vector<double>& y)  {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::computeAllRatesOfChange()                             {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::evalModel(double time, vector<double>& y)             {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
//void  IModel::evalEvents(double time, vector<double>& y){}
void  IModel::resetEvents()                                         {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::evalInitialAssignments()                              {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::testConstraints()                                     {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}
void  IModel::InitializeRateRuleSymbols()                           {Log(lError) << "Called un implemented function "<<__FUNCTION__<<" in IModel!!";}

} //namespace rr
