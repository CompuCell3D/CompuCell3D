#ifndef rrRoadRunnerH
#define rrRoadRunnerH

#include <string>
#include "rr-libstruct/lsMatrix.h"
#include "rr-libstruct/lsLibStructural.h"

#include "rrObject.h"
#include "rrTVariableType.h"
#include "rrTParameterType.h"
#include "rrCVODEInterface.h"
#include "rrNLEQInterface.h"
#include "rrStringList.h"
#include "rrStringListContainer.h"
#include "rrMisc.h"
#include "rrTextWriter.h"
#include "rrSimulationData.h"
#include "rrSimulationSettings.h"
#include "rrCompiler.h"
#include "rrArrayList.h"
#include "rrArrayList2.h"
#include "rrXMLDocument.h"
#include "rrNOMSupport.h"
#include "rrConstants.h"
#include "rrNewArrayList.h"
#include "rrPluginManager.h"
#include "rrModelSharedLibrary.h"
#include "Poco/Thread.h"
#include <map>

namespace rr
{
using Poco::Mutex;
using std::string;
using namespace ls;

class ModelGenerator;
class SBMLModelSimulation;
class ModelFromC;
class CSharpGenerator;
class CGenerator;

class RR_DECLSPEC RoadRunner : public rrObject
{
	private:
    	static int	  					mInstanceCount;
    	int	  							mInstanceID;
		bool                            mUseKinsol;
		const double                    mDiffStepSize;

        const string					mModelFolder;			//Folder for XML models
		const double                    mSteadyStateThreshold;
        DoubleMatrix                    mRawSimulationData;
		SimulationData                  mSimulationData;
	    string 							mSupportCodeFolder;		//The compiler needs this in order to compile models

		string                   		mTempFileFolder;
        string							mCurrentSBMLFileName;
		SBMLModelSimulation            *mSimulation;

		CvodeInterface                 *mCVode;
		ISteadyStateSolver             *mSteadyStateSolver;
		vector<TSelectionRecord>        mSelectionList;
		ModelGenerator                 *mModelGenerator;    //Pointer to one of the below ones..
		CSharpGenerator                *mCSharpGenerator;
		CGenerator                     *mCGenerator;
		Compiler                        mCompiler;

		bool                     		mComputeAndAssignConservationLaws;

		vector<TSelectionRecord>        mSteadyStateSelection;
		double                          mTimeStart;
		double                          mTimeEnd;
		int                             mNumPoints;

   		static Mutex	 				mLibSBMLMutex;
   		static Mutex	 				mCompileMutex;
		ModelFromC*                     mModel;
		ModelSharedLibrary	  	  		mModelLib;
		string                          mCurrentSBML;
		LibStructural                   mLS;                                //libstruct library

		SimulationSettings              mSettings;
		NOMSupport						mNOM;
		PluginManager					mPluginManager;

		//std::map<std::string,double>    exportableQuantities; // this is a map of floatingSpecies boundary species that could be exported as a dictionary to e.g. Python wrapper fcn 


		void                            addNthOutputToResult(DoubleMatrix& results, int nRow, double dCurrentTime);
		bool                            populateResult();
		bool                            isNleqAvailable();

		double                          getValueForRecord(const TSelectionRecord& record);
		double                          getNthSelectedOutput(const int& index, const double& dCurrentTime);
		vector<double>                  buildModelEvalArgument();
		double                          getVariableValue(const TVariableType& variableType, const int& variableIndex);


		vector<TSelectionRecord>        getSteadyStateSelection(const StringList& newSelectionList);
		StringList                      getParameterIds();
		bool 							loadSBMLIntoNOM(const string& sbml);
		bool 							loadSBMLIntoLibStruct(const string& sbml);
		string 							createModelName(const string& mCurrentSBMLFileName);
		

	public: //These should be hidden later on...
		bool                     		mConservedTotalChanged;

	public:
 										RoadRunner(const string& tempFolder = gDefaultTempFolder, const string& supportCodeFolder = gDefaultSupportCodeFolder, const string& compiler = gDefaultCompiler);
		virtual                        ~RoadRunner();
        int								getInstanceID();
        int								getInstanceCount();

		bool        					computeAndAssignConservationLaws();
		void                            setParameterValue(const TParameterType& parameterType, const int& parameterIndex, const double& value);
		double                          getParameterValue(const TParameterType& parameterType, const int& parameterIndex);

		string                  		getParamPromotedSBML(const string& sArg);
		NOMSupport*						getNOM();
		string							getInfo();
        PluginManager&					getPluginManager();

		//Compiling
   		Compiler*						getCompiler();
        bool							compileSource(const string& modelSourceCodeName);
		//Functions --------------------------------------------------------------------
        bool                            isModelLoaded();
        bool                            setCompiler(const string& compiler);
        string                          getModelName();
		string							getlibSBMLVersion();
        bool                            unLoadModel();
        bool                            unLoadModelDLL();
        CvodeInterface*                 getCVodeInterface();
        NLEQInterface*                  getNLEQInterface();
        int 							createDefaultSteadyStateSelectionList();
        int                             createDefaultTimeCourseSelectionList();
		int                             createTimeCourseSelectionList();
		bool                     		setTempFileFolder(const string& folder);
		string                   		getTempFolder();

		//Simulation stuff
		DoubleMatrix                    simulate();
		bool                            simulate2();
		DoubleMatrix                    simulateEx(const double& startTime, const double& endTime, const int& numberOfPoints);
		void                            partOfSimulation(SBMLModelSimulation* simulation){mSimulation = simulation;}
		SimulationData                  getSimulationResult();
        bool							loadSimulationSettings(const string& fName);
		bool                            useSimulationSettings(SimulationSettings& settings);
		DoubleMatrix                    runSimulation();
		bool                            initializeModel();
		bool                            simulateSBMLFile(const string& fileName, const bool& useConservationLaws);
		bool 							createDefaultSelectionLists();
		string                          getSBML();
		double                          getTimeStart();
		double                          getTimeEnd();
		int                             getNumPoints();
		void                            setTimeStart(const double& startTime);
		void                            setTimeEnd(const double& endTime);
		void                            setNumPoints(const int& nummberOfPoints);
		void                            reset();
		void                            changeInitialConditions(const vector<double>& ic);

		std::map<std::string,double>    getFloatingSpeciesMap(); 
		void							setFloatingSpeciesMap(const std::map<std::string,double> & _speciesMap); 

		std::map<std::string,double>    getAdjustableSBMLParameters(); 
		void							setAdjustableSBMLParameters(const std::map<std::string,double> & _speciesMap); 

		//Model generation
		ModelFromC*						getModel();
		CGenerator*						getCGenerator();
		CSharpGenerator*				getCSharpGenerator();
		void                            resetModelGenerator();
		bool                            compileModel();
		bool                            compileCurrentModel();
		ModelFromC*                     createModel();
		bool                            generateModelCode(const string& sbml = gEmptyString, const string& baseName = gEmptyString);
		ModelGenerator*                 getCodeGenerator();
		string                          getCHeaderCode();
		string                          getCSourceCode();
		string                          getCSharpCode();	
		bool                            loadSBMLFromFile(const string& fileName, const bool& forceReCompile = false);
		bool                            loadSBML(const string& sbml, const bool& forceReCompile = false);

		vector<double>                  getReactionRates();
		vector<double>                  getRatesOfChange();
		StringList                      getSpeciesIds();
		StringList                      getReactionIds();

 		// ---------------------------------------------------------------------
		// Start of Level 2 API Methods
		// ---------------------------------------------------------------------
		string                          getCapabilities();
		void                            setTolerances(const double& aTol, const double& rTol);
		void                            setTolerances(const double& aTol, const double& rTol, const int& maxSteps);
		void                            correctMaxStep();
		void                            setCapabilities(const string& capsStr);
		bool                            setValue(const string& sId, const double& dValue);
		double                          getValue(const string& sId);
		NewArrayList                    getAvailableTimeCourseSymbols();
		StringList                      getTimeCourseSelectionList();
		void                            setTimeCourseSelectionList(const string& List);
		void                            setTimeCourseSelectionList(const StringList& newSelectionList);
		double                          oneStep(const double& currentTime, const double& stepSize);
		double                          oneStep(const double& currentTime, const double& stepSize, const bool& reset);

		// ---------------------------------------------------------------------
		// Start of Level 3 API Methods
		// ---------------------------------------------------------------------
		double                          steadyState();
		DoubleMatrix                    getFullJacobian();
		DoubleMatrix 					getFullReorderedJacobian();
		DoubleMatrix                    getReducedJacobian();
        DoubleMatrix                    getEigenvalues();
		DoubleMatrix                    getEigenvaluesFromMatrix (DoubleMatrix m);
		vector<Complex>                 getEigenvaluesCpx();

		// ---------------------------------------------------------------------
		// Start of Level 4 API Methods
		// ---------------------------------------------------------------------
		DoubleMatrix*                   getLinkMatrix();
		DoubleMatrix*                   getNrMatrix();
		DoubleMatrix*                   getL0Matrix();
		DoubleMatrix                    getStoichiometryMatrix();
		DoubleMatrix                    getReorderedStoichiometryMatrix();
		DoubleMatrix                    getFullyReorderedStoichiometryMatrix();
 		DoubleMatrix                    getConservationMatrix();
		DoubleMatrix                    getUnscaledConcentrationControlCoefficientMatrix();
		DoubleMatrix                    getScaledConcentrationControlCoefficientMatrix();
        DoubleMatrix                    getUnscaledFluxControlCoefficientMatrix();
        DoubleMatrix                    getScaledFluxControlCoefficientMatrix();
		int                             getNumberOfDependentSpecies();
		int                             getNumberOfIndependentSpecies();
		void                            computeContinuation(const double& stepSize, const int& independentVariable, const string& parameterTypeStr);
        NewArrayList                 	getUnscaledFluxControlCoefficientIds();
		NewArrayList           	      	getFluxControlCoefficientIds();
        NewArrayList                 	getUnscaledConcentrationControlCoefficientIds();
		NewArrayList                 	getConcentrationControlCoefficientIds();
		NewArrayList                 	getElasticityCoefficientIds();
		NewArrayList                 	getUnscaledElasticityCoefficientIds();
		StringList                      getEigenvalueIds();
		NewArrayList                 	getAvailableSteadyStateSymbols();
		StringList                   	getSteadyStateSelectionList();
		void                            setSteadyStateSelectionList(const StringList& newSelectionList);
		double                          computeSteadyStateValue(const TSelectionRecord& record);
		vector<double>                  computeSteadyStateValues();
		vector<double>                  computeSteadyStateValues(const StringList& selection);
		vector<double>                  computeSteadyStateValues(const vector<TSelectionRecord>& selection, const bool& computeSteadyState);
		double                          computeSteadyStateValue(const string& sId);
		vector<double>                  getSelectedValues();

        //		void                     		reMultiplyCompartments(const bool& bValue);

		void                     		computeAndAssignConservationLaws(const bool& bValue);
		double*                         steadyStateParameterScan(const string& symbol, const double& startValue, const double& endValue, const double& stepSize);
		string                          writeSBML();
		int                             getNumberOfLocalParameters(const int& reactionId);
		void                            setLocalParameterByIndex(const int& reactionId, const int index, const double& value);
		double                          getLocalParameterByIndex(const int& reactionId, const int& index);
		void                            setLocalParameterValues(const int& reactionId, const vector<double>& values);
		vector<double>                  getLocalParameterValues(const int& reactionId);
		StringList                      getLocalParameterIds(const int& reactionId);
		StringList                      getAllLocalParameterTupleList();
		int                             getNumberOfReactions();
		double                          getReactionRate(const int& index);
		double                          getRateOfChange(const int& index);
		StringList                      getRateOfChangeIds();
		vector<double>                  getRatesOfChangeEx(const vector<double>& values);
		vector<double>                  getReactionRatesEx(const vector<double>& values);
		vector<string>                  getFloatingSpeciesIdsArray();
		vector<string>                  getGlobalParameterIdsArray();
		int                             getNumberOfCompartments();
		void                            setCompartmentByIndex(const int& index, const double& value);
		double                          getCompartmentByIndex(const int& index);
		void                            setCompartmentVolumes(const vector<double>& values);
		StringList                      getCompartmentIds();
		int                             getNumberOfBoundarySpecies();
		void                            setBoundarySpeciesByIndex(const int& index, const double& value);
		double                          getBoundarySpeciesByIndex(const int& index);
		vector<double>                  getBoundarySpeciesConcentrations();
		void                            setBoundarySpeciesConcentrations(const vector<double>& values);
		StringList                      getBoundarySpeciesIds();
		StringList                      getBoundarySpeciesAmountIds();
		int                             getNumberOfFloatingSpecies();
		void                            setFloatingSpeciesByIndex(const int& index, const double& value);
		double                          getFloatingSpeciesByIndex(const int& index);
		vector<double>                  getFloatingSpeciesConcentrations();
		vector<double>                  getFloatingSpeciesInitialConcentrations();
		void                            setFloatingSpeciesConcentrations(const vector<double>& values);
		void                            setFloatingSpeciesInitialConcentrationByIndex(const int& index, const double& value);
		void                            setFloatingSpeciesInitialConcentrations(const vector<double>& values);
		StringList                      getFloatingSpeciesIds();
		StringList                      getFloatingSpeciesInitialConditionIds();
		StringList                      getFloatingSpeciesAmountIds();
		int                             getNumberOfGlobalParameters();
		void                            setGlobalParameterByIndex(const int& index, const double& value);
		double                          getGlobalParameterByIndex(const int& index);
		void                            setGlobalParameterValues(const vector<double>& values);
		vector<double>                  getGlobalParameterValues();
		StringList                      getGlobalParameterIds();
		StringList                      getAllGlobalParameterTupleList();
		void                            evalModel();

		//These functions are better placed in a separate file, as non class members, but part of the roadrunner namespace?
		string                          getName();
		string                          getVersion();
		string                          getAuthor();
		string                          getDescription();
		string                          getDisplayName();
		string                          getCopyright();
		string                          getURL();

		//Plugin stuff
        bool							loadPlugins();
        bool							unLoadPlugins();

		//RoadRunner MCA functions......

        //[Help("Get unscaled control coefficient with respect to a global parameter")]
		double 							getuCC(const string& variableName, const string& parameterName);

		//[Help("Get scaled control coefficient with respect to a global parameter")]
		double 							getCC(const string& variableName, const string& parameterName);

		//[Help("Get unscaled elasticity coefficient with respect to a global parameter or species")]
		double 							getuEE(const string& reactionName, const string& parameterName);

		//[Help("Get unscaled elasticity coefficient with respect to a global parameter or species. Optionally the model is brought to steady state after the computation.")]
		double 							getuEE(const string& reactionName, const string& parameterName, bool computeSteadystate);

    	//[Help("Get scaled elasticity coefficient with respect to a global parameter or species")]
		double 							getEE(const string& reactionName, const string& parameterName);

		//[Help("Get scaled elasticity coefficient with respect to a global parameter or species. Optionally the model is brought to steady state after the computation.")]
		double 							getEE(const string& reactionName, const string& parameterName, bool computeSteadyState);

        // Get a single species elasticity value
        // IMPORTANT:
        // Assumes that the reaction rates have been precomputed at the operating point !!
		double 							getUnscaledSpeciesElasticity(int reactionId, int speciesIndex);

		//"Returns the elasticity of a given reaction to a given parameter. Parameters can be boundary species or global parameters"
		double 							getUnScaledElasticity(const string& reactionName, const string& parameterName);

		//"Compute the unscaled species elasticity matrix at the current operating point")]
		DoubleMatrix 					getUnscaledElasticityMatrix();

		//"Compute the unscaled elasticity matrix at the current operating point")]
		DoubleMatrix 					getScaledReorderedElasticityMatrix();

		//[Help("Compute the unscaled elasticity for a given reaction and given species")]
		double 							getUnscaledFloatingSpeciesElasticity(const string& reactionName, const string& speciesName);

		//[Help("Compute the scaled elasticity for a given reaction and given species")]
		double 							getScaledFloatingSpeciesElasticity(const string& reactionName, const string& speciesName);

		// Changes a given parameter type by the given increment
		void 							changeParameter(TParameterType parameterType, int reactionIndex, int parameterIndex, double originalValue, double increment);

        //[Help("Returns the unscaled elasticity for a named reaction with respect to a named parameter (local or global)"
        double 							getUnscaledParameterElasticity(const string& reactionName, const string& parameterName);

        //"Compute the value for a particular unscaled concentration control coefficients with respect to a local parameter"
        double 							getUnscaledConcentrationControlCoefficient(const string& speciesName, const string& localReactionName, const string& parameterName);

        //"Compute the value for a particular scaled concentration control coefficients with respect to a local parameter"
        double 							getScaledConcentrationControlCoefficient(const string& speciesName, const string& localReactionName, const string& parameterName);

        //"Compute the value for a particular concentration control coefficient, permitted parameters include global parameters, boundary conditions and conservation totals"
        double 							getUnscaledConcentrationControlCoefficient(const string& speciesName, const string& parameterName);

        //"Compute the value for a particular scaled concentration control coefficients with respect to a global or boundary species parameter"
        double 							getScaledConcentrationControlCoefficient(const string& speciesName, const string& parameterName);

		//[Help("Compute the value for a particular unscaled flux control coefficients with respect to a local parameter")
        double 							getUnscaledFluxControlCoefficient(const string& fluxName, const string& localReactionName, const string& parameterName);

        //"Compute the value for a particular flux control coefficient, permitted parameters include global parameters, boundary conditions and conservation totals"
        double 							getUnscaledFluxControlCoefficient(const string& reactionName, const string& parameterName);

        //[Help("Compute the value for a particular scaled flux control coefficients with respect to a local parameter");]
        double 							getScaledFluxControlCoefficient(const string& reactionName, const string& localReactionName, const string& parameterName);

        //    "Compute the value for a particular scaled flux control coefficients with respect to a global or boundary species parameter"
        double 							getScaledFluxControlCoefficient(const string& reactionName, const string& parameterName);

};

}

#endif

/*! \mainpage RoadRunner C++ Library

\par
This document describes the application programming interface (API) of RoadRunner, an open source (BSD) library for computing structural characteristics of cellular networks.
\par

\par Dependencies
The RoadRunner library depend on several third-party libraries, CLapack, libSBML, Sundials, NLEQ, Poco and Pugi. These are provided with the binary installation where necessary.
\par

\author		Totte Karlsson (totte@dunescientific.com)
\author  	Frank T. Bergmann (fbergman@u.washington.edu)
\author     Herbert M. Sauro  (hsauro@u.washington.edu)

\par License
\par
Copyright (c) 2012, Frank T Bergmann and Herbert M Sauro\n
All rights reserved.

\par
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

\li Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

\li Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

\li Neither the name of University of Washington nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

\par

*/


////C# - four slashes to make it clearer...
//// Below are first RoadRunner MCA - partial class and then RoadRunner.cs

////using System;
////using System.Collections;
////using System.Diagnostics;
////using LibRoadRunner.Util;
////using LibRoadRunner.Util.Unused;
////using libstructural;
////using SBW;
////
////namespace LibRoadRunner
////{
////    partial class RoadRunner
////    {
////        [Help("Get unscaled control coefficient with respect to a global parameter")]
////        public double getuCC(string variableName, string parameterName)
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    TParameterType parameterType;
////                    TVariableType variableType;
////                    double originalParameterValue;
////                    int variableIndex;
////                    int parameterIndex;
////                    double f1;
////                    double f2;
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    // Check the variable name
////                    if (ModelGenerator.Instance.reactionList.find(variableName, out variableIndex))
////                    {
////                        variableType = TVariableType.vtFlux;
////                    }
////                    else if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(variableName,
////                                                                                           out variableIndex))
////                    {
////                        variableType = TVariableType.vtSpecies;
////                    }
////                    else throw new SBWApplicationException("Unable to locate variable: [" + variableName + "]");
////
////                    // Check for the parameter name
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptGlobalParameter;
////                        originalParameterValue = model.gp[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptBoundaryParameter;
////                        originalParameterValue = model.bc[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptConservationParameter;
////                        originalParameterValue = model.ct[parameterIndex];
////                    }
////                    else throw new SBWApplicationException("Unable to locate parameter: [" + parameterName + "]");
////
////                    // Get the original parameter value
////                    originalParameterValue = getParameterValue(parameterType, parameterIndex);
////
////                    double hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        model.convertToConcentrations();
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi = getVariableValue(variableType, variableIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi2 = getVariableValue(variableType, variableIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd = getVariableValue(variableType, variableIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd2 = getVariableValue(variableType, variableIndex);
////
////                        // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        // What ever happens, make sure we restore the parameter level
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue);
////                        steadyState();
////                    }
////                    return 1/(12*hstep)*(f1 + f2);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getuCC ()", e.Message);
////            }
////        }
////
////
////        [Help("Get scaled control coefficient with respect to a global parameter")]
////        public double getCC(string variableName, string parameterName)
////        {
////            TVariableType variableType;
////            TParameterType parameterType;
////            int variableIndex;
////            int parameterIndex;
////            //double originalParameterValue;
////
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            // Check the variable name
////            if (ModelGenerator.Instance.reactionList.find(variableName, out variableIndex))
////            {
////                variableType = TVariableType.vtFlux;
////            }
////            else if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(variableName, out variableIndex))
////            {
////                variableType = TVariableType.vtSpecies;
////            }
////            else throw new SBWApplicationException("Unable to locate variable: [" + variableName + "]");
////
////            // Check for the parameter name
////            if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptGlobalParameter;
////                //originalParameterValue = model.gp[parameterIndex];
////            }
////            else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptBoundaryParameter;
////                //originalParameterValue = model.bc[parameterIndex];
////            }
////            else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptConservationParameter;
////                //originalParameterValue = model.ct[parameterIndex];
////            }
////
////
////            else throw new SBWApplicationException("Unable to locate parameter: [" + parameterName + "]");
////
////            steadyState();
////            double variableValue = getVariableValue(variableType, variableIndex);
////            double parameterValue = getParameterValue(parameterType, parameterIndex);
////
////            return getuCC(variableName, parameterName)*parameterValue/variableValue;
////        }
////
////
////        [Help("Get unscaled elasticity coefficient with respect to a global parameter or species")]
////        public double getuEE(string reactionName, string parameterName)
////        {
////            return getuEE(reactionName, parameterName, true);
////        }
////
////        [Help("Get unscaled elasticity coefficient with respect to a global parameter or species. Optionally the model is brought to steady state after the computation.")]
////        public double getuEE(string reactionName, string parameterName, bool computeSteadystate)
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    TParameterType parameterType;
////                    double originalParameterValue;
////                    int reactionIndex;
////                    int parameterIndex;
////                    double f1;
////                    double f2;
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    // Check the reaction name
////                    if (!ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex))
////                    {
////                        throw new SBWApplicationException("Unable to locate reaction name: [" + reactionName + "]");
////                    }
////
////                    // Find out what kind of parameter we are dealing with
////                    if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptFloatingSpecies;
////                        originalParameterValue = model.y[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptBoundaryParameter;
////                        originalParameterValue = model.bc[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptGlobalParameter;
////                        originalParameterValue = model.gp[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptConservationParameter;
////                        originalParameterValue = model.ct[parameterIndex];
////                    }
////                    else throw new SBWApplicationException("Unable to locate variable: [" + parameterName + "]");
////
////                    double hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        model.convertToConcentrations();
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
////                        model.computeReactionRates(model.time, model.y);
////                        double fi = model.rates[reactionIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
////                        model.computeReactionRates(model.time, model.y);
////                        double fi2 = model.rates[reactionIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
////                        model.computeReactionRates(model.time, model.y);
////                        double fd = model.rates[reactionIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
////                        model.computeReactionRates(model.time, model.y);
////                        double fd2 = model.rates[reactionIndex];
////
////                        // Use instead the 5th order approximation double unscaledValue = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        // What ever happens, make sure we restore the parameter level
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue);
////                        model.computeReactionRates(model.time, model.y);
////                        if (computeSteadystate) steadyState();
////                    }
////                    return 1/(12*hstep)*(f1 + f2);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getuEE ()", e.Message);
////            }
////        }
////
////
////        [Help("Get scaled elasticity coefficient with respect to a global parameter or species")]
////        public double getEE(string reactionName, string parameterName)
////        {
////            return getEE(reactionName, parameterName, true);
////        }
////
////        [Help("Get scaled elasticity coefficient with respect to a global parameter or species. Optionally the model is brought to steady state after the computation.")]
////        public double getEE(string reactionName, string parameterName, bool computeSteadyState)
////        {
////            TParameterType parameterType;
////            int reactionIndex;
////            int parameterIndex;
////
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            // Check the reaction name
////            if (!ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex))
////            {
////                throw new SBWApplicationException(String.Format("Unable to locate reaction name: [{0}]", reactionName));
////            }
////
////            // Find out what kind of parameter we are dealing with
////            if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptFloatingSpecies;
////            }
////            else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptBoundaryParameter;
////            }
////            else if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptGlobalParameter;
////            }
////            else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////            {
////                parameterType = TParameterType.ptConservationParameter;
////            }
////            else throw new SBWApplicationException(String.Format("Unable to locate variable: [{0}]", parameterName));
////
////            model.computeReactionRates(model.time, model.y);
////            double variableValue = model.rates[reactionIndex];
////            double parameterValue = getParameterValue(parameterType, parameterIndex);
////            if (variableValue == 0) variableValue = 1e-12;
////            return getuEE(reactionName, parameterName, computeSteadyState) * parameterValue / variableValue;
////        }
////
////        [Ignore]
////        // Get a single species elasticity value
////        // IMPORTANT:
////        // Assumes that the reaction rates have been precomputed at the operating point !!
////        private double getUnscaledSpeciesElasticity(int reactionId, int speciesIndex)
////        {
////            double f1, f2, fi, fi2, fd, fd2;
////            double originalParameterValue = model.getConcentration(speciesIndex);
////
////            double hstep = DiffStepSize*originalParameterValue;
////            if (Math.Abs(hstep) < 1E-12)
////                hstep = DiffStepSize;
////
////            model.convertToConcentrations();
////            model.setConcentration(speciesIndex, originalParameterValue + hstep);
////            try
////            {
////                model.computeReactionRates(model.time, model.y);
////                fi = model.rates[reactionId];
////
////                model.setConcentration(speciesIndex, originalParameterValue + 2*hstep);
////                model.computeReactionRates(model.time, model.y);
////                fi2 = model.rates[reactionId];
////
////                model.setConcentration(speciesIndex, originalParameterValue - hstep);
////                model.computeReactionRates(model.time, model.y);
////                fd = model.rates[reactionId];
////
////                model.setConcentration(speciesIndex, originalParameterValue - 2*hstep);
////                model.computeReactionRates(model.time, model.y);
////                fd2 = model.rates[reactionId];
////
////                // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                // The following separated lines avoid small amounts of roundoff error
////                f1 = fd2 + 8*fi;
////                f2 = -(8*fd + fi2);
////            }
////            finally
////            {
////                // What ever happens, make sure we restore the species level
////                model.setConcentration(speciesIndex, originalParameterValue);
////            }
////            return 1/(12*hstep)*(f1 + f2);
////        }
////
////
////        [Help(
////            "Returns the elasticity of a given reaction to a given parameter. Parameters can be boundary species or global parameters"
////            )]
////        public double getUnScaledElasticity(string reactionName, string parameterName)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            double f1, f2, fi, fi2, fd, fd2;
////            double hstep;
////
////            int reactionId = -1;
////            if (!(ModelGenerator.Instance.reactionList.find(reactionName, out reactionId)))
////                throw new SBWApplicationException("Unrecognized reaction name in call to getUnScaledElasticity [" +
////                                                  reactionName + "]");
////
////            int index = -1;
////            // Find out what kind of parameter it is, species or global parmaeter
////            if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out index))
////            {
////                double originalParameterValue = model.bc[index];
////                hstep = DiffStepSize*originalParameterValue;
////                if (Math.Abs(hstep) < 1E-12)
////                    hstep = DiffStepSize;
////
////                try
////                {
////                    model.convertToConcentrations();
////                    model.bc[index] = originalParameterValue + hstep;
////                    model.computeReactionRates(model.time, model.y);
////                    fi = model.rates[reactionId];
////
////                    model.bc[index] = originalParameterValue + 2*hstep;
////                    model.computeReactionRates(model.time, model.y);
////                    fi2 = model.rates[reactionId];
////
////                    model.bc[index] = originalParameterValue - hstep;
////                    model.computeReactionRates(model.time, model.y);
////                    fd = model.rates[reactionId];
////
////                    model.bc[index] = originalParameterValue - 2*hstep;
////                    model.computeReactionRates(model.time, model.y);
////                    fd2 = model.rates[reactionId];
////
////                    // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                    // The following separated lines avoid small amounts of roundoff error
////                    f1 = fd2 + 8*fi;
////                    f2 = -(8*fd + fi2);
////                }
////                finally
////                {
////                    model.bc[index] = originalParameterValue;
////                }
////            }
////            else
////            {
////                if (ModelGenerator.Instance.globalParameterList.find(parameterName, out index))
////                {
////                    double originalParameterValue = model.gp[index];
////                    hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        model.convertToConcentrations();
////
////                        model.gp[index] = originalParameterValue + hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fi = model.rates[reactionId];
////
////                        model.gp[index] = originalParameterValue + 2*hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fi2 = model.rates[reactionId];
////
////                        model.gp[index] = originalParameterValue - hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fd = model.rates[reactionId];
////
////                        model.gp[index] = originalParameterValue - 2*hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fd2 = model.rates[reactionId];
////
////                        // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        model.gp[index] = originalParameterValue;
////                    }
////                }
////                else if (ModelGenerator.Instance.conservationList.find(parameterName, out index))
////                {
////                    double originalParameterValue = model.gp[index];
////                    hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        model.convertToConcentrations();
////
////                        model.ct[index] = originalParameterValue + hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fi = model.rates[reactionId];
////
////                        model.ct[index] = originalParameterValue + 2*hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fi2 = model.rates[reactionId];
////
////                        model.ct[index] = originalParameterValue - hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fd = model.rates[reactionId];
////
////                        model.ct[index] = originalParameterValue - 2*hstep;
////                        model.computeReactionRates(model.time, model.y);
////                        fd2 = model.rates[reactionId];
////
////                        // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        model.ct[index] = originalParameterValue;
////                    }
////                }
////                else
////                    throw new SBWApplicationException("Unrecognized parameter name in call to getUnScaledElasticity [" +
////                                                      parameterName + "]");
////            }
////            return 1/(12*hstep)*(f1 + f2);
////        }
////
////
////        [Help("Compute the unscaled species elasticity matrix at the current operating point")]
////        public double[][] getUnscaledElasticityMatrix()
////        {
////            var uElastMatrix = new double[model.getNumReactions][];
////            for (int i = 0; i < model.getNumReactions; i++) uElastMatrix[i] = new double[model.getNumTotalVariables];
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    model.convertToConcentrations();
////                    // Compute reaction velocities at the current operating point
////                    model.computeReactionRates(model.time, model.y);
////
////                    for (int i = 0; i < model.getNumReactions; i++)
////                        for (int j = 0; j < model.getNumTotalVariables; j++)
////                            uElastMatrix[i][j] = getUnscaledSpeciesElasticity(i, j);
////
////                    return uElastMatrix;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from unscaledElasticityMatrix()", e.Message);
////            }
////        }
////
////        [Help("Compute the unscaled elasticity matrix at the current operating point")]
////        public double[][] getScaledElasticityMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] uelast = getUnscaledElasticityMatrix();
////
////                    var result = new double[uelast.Length][];
////                    for (int i = 0; i < uelast.Length; i++)
////                        result[i] = new double[uelast[0].Length];
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////                    double[] rates = model.rates;
////                    for (int i = 0; i < uelast.Length; i++)
////                    {
////                        // Rows are rates
////                        if (rates[i] == 0)
////                            throw new SBWApplicationException("Unable to compute elasticity, reaction rate [" +
////                                                              ModelGenerator.Instance.reactionList[i].name +
////                                                              "] set to zero");
////
////                        for (int j = 0; j < uelast[0].Length; j++) // Columns are species
////                            result[i][j] = uelast[i][j]*model.getConcentration(j)/rates[i];
////                    }
////                    return result;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from scaledElasticityMatrix()", e.Message);
////            }
////        }
////
////
////        [Help("Compute the unscaled elasticity for a given reaction and given species")]
////        public double getUnscaledFloatingSpeciesElasticity(string reactionName, string speciesName)
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    int speciesIndex = 0;
////                    int reactionIndex = 0;
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.floatingSpeciesConcentrationList.find(speciesName, out speciesIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate species name while computing unscaled elasticity");
////                    if (!ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate reaction name while computing unscaled elasticity");
////
////                    return getUnscaledSpeciesElasticity(reactionIndex, speciesIndex);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from scaledElasticityMatrix()", e.Message);
////            }
////        }
////
////        [Help("Compute the scaled elasticity for a given reaction and given species")]
////        public double getScaledFloatingSpeciesElasticity(string reactionName, string speciesName)
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    int speciesIndex = 0;
////                    int reactionIndex = 0;
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.floatingSpeciesConcentrationList.find(speciesName, out speciesIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate species name while computing unscaled elasticity");
////                    if (!ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate reaction name while computing unscaled elasticity");
////
////                    return getUnscaledSpeciesElasticity(reactionIndex, speciesIndex)*
////                           model.getConcentration(speciesIndex)/model.rates[reactionIndex];
////                    ;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from scaledElasticityMatrix()", e.Message);
////            }
////        }
////
////
////        [Ignore]
////        // Changes a given parameter type by the given increment
////        private void changeParameter(TParameterType parameterType, int reactionIndex, int parameterIndex,
////                                     double originalValue, double increment)
////        {
////            switch (parameterType)
////            {
////                case TParameterType.ptLocalParameter:
////                    model.lp[reactionIndex][parameterIndex] = originalValue + increment;
////                    break;
////                case TParameterType.ptGlobalParameter:
////                    model.gp[parameterIndex] = originalValue + increment;
////                    break;
////                case TParameterType.ptBoundaryParameter:
////                    model.bc[parameterIndex] = originalValue + increment;
////                    break;
////                case TParameterType.ptConservationParameter:
////                    model.ct[parameterIndex] = originalValue + increment;
////                    break;
////            }
////        }
////
////
////        [Help("Returns the unscaled elasticity for a named reaction with respect to a named parameter (local or global)"
////            )]
////        private double getUnscaledParameterElasticity(string reactionName, string parameterName)
////        {
////            int reactionIndex;
////            int parameterIndex;
////            double originalParameterValue;
////            TParameterType parameterType;
////
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            model.convertToConcentrations();
////            model.computeReactionRates(model.time, model.y);
////
////            if (!ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex))
////                throw new SBWApplicationException(
////                    "Internal Error: unable to locate reaction name while computing unscaled elasticity");
////
////            // Look for the parameter name, check local parameters first, then global
////            if (ModelGenerator.Instance.localParameterList[reactionIndex].find(reactionName, parameterName,
////                                                                               out parameterIndex))
////                parameterType = TParameterType.ptLocalParameter;
////            else if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                parameterType = TParameterType.ptGlobalParameter;
////            else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                parameterType = TParameterType.ptBoundaryParameter;
////            else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                parameterType = TParameterType.ptConservationParameter;
////            else
////                return 0.0;
////
////            double f1, f2, fi, fi2, fd, fd2;
////            originalParameterValue = 0.0;
////            switch (parameterType)
////            {
////                case TParameterType.ptLocalParameter:
////                    originalParameterValue = model.lp[reactionIndex][parameterIndex];
////                    break;
////                case TParameterType.ptGlobalParameter:
////                    originalParameterValue = model.gp[parameterIndex];
////                    break;
////                case TParameterType.ptBoundaryParameter:
////                    originalParameterValue = model.bc[parameterIndex];
////                    break;
////                case TParameterType.ptConservationParameter:
////                    originalParameterValue = model.ct[parameterIndex];
////                    break;
////            }
////
////            double hstep = DiffStepSize*originalParameterValue;
////            if (Math.Abs(hstep) < 1E-12)
////                hstep = DiffStepSize;
////
////            try
////            {
////                changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, hstep);
////                model.convertToConcentrations();
////                model.computeReactionRates(model.time, model.y);
////                fi = model.rates[reactionIndex];
////
////                changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, 2*hstep);
////                model.computeReactionRates(model.time, model.y);
////                fi2 = model.rates[reactionIndex];
////
////                changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -hstep);
////                model.computeReactionRates(model.time, model.y);
////                fd = model.rates[reactionIndex];
////
////                changeParameter(parameterType, reactionIndex, parameterIndex, originalParameterValue, -2*hstep);
////                model.computeReactionRates(model.time, model.y);
////                fd2 = model.rates[reactionIndex];
////
////                // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                // The following separated lines avoid small amounts of roundoff error
////                f1 = fd2 + 8*fi;
////                f2 = -(8*fd + fi2);
////            }
////            finally
////            {
////                // What ever happens, make sure we restore the species level
////                switch (parameterType)
////                {
////                    case TParameterType.ptLocalParameter:
////                        model.lp[reactionIndex][parameterIndex] = originalParameterValue;
////                        break;
////                    case TParameterType.ptGlobalParameter:
////                        model.gp[parameterIndex] = originalParameterValue;
////                        break;
////                    case TParameterType.ptBoundaryParameter:
////                        model.bc[parameterIndex] = originalParameterValue;
////                        break;
////                    case TParameterType.ptConservationParameter:
////                        model.ct[parameterIndex] = originalParameterValue;
////                        break;
////                }
////            }
////            return 1/(12*hstep)*(f1 + f2);
////        }
////
////
////        // Use the formula: ucc = -L Jac^-1 Nr
////        [Help("Compute the matrix of unscaled concentration control coefficients")]
////        public double[][] getUnscaledConcentrationControlCoefficientMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    Matrix uelast;
////                    Matrix Nr;
////                    Matrix LinkMatrix;
////
////                    setTimeStart(0.0);
////                    setTimeEnd(50.0);
////                    setNumPoints(1);
////                    simulate();
////                    if (steadyState() > STEADYSTATE_THRESHOLD)
////                    {
////                        if (steadyState() > 1E-2)
////                            throw new SBWApplicationException(
////                                "Unable to locate steady state during frequency response computation");
////                    }
////
////                    uelast = new Matrix(getUnscaledElasticityMatrix());
////                    Nr = new Matrix(getNrMatrix());
////                    LinkMatrix = new Matrix(getLinkMatrix());
////
////                    var Inv = new Matrix(Nr.nRows, LinkMatrix.nCols);
////                    var T2 = new Matrix(Nr.nRows, LinkMatrix.nCols); // Stores -Jac  and (-Jac)^-1
////                    var T3 = new Matrix(LinkMatrix.nRows, 1); // Stores (-Jac)^-1 . Nr
////                    var T4 = new Matrix(Nr.nRows, Nr.nCols);
////
////                    // Compute the Jacobian first
////                    var T1 = new Matrix(Nr.nRows, uelast.nCols);
////                    T1.mult(Nr, uelast);
////                    var Jac = new Matrix(Nr.nRows, LinkMatrix.nCols);
////                    Jac.mult(T1, LinkMatrix);
////                    T2.mult(Jac, -1.0); // Compute -Jac
////
////                    //ArrayList reactionNames = getReactionNames();
////                    //ArrayList speciesNames = getSpeciesNames();
////
////                    //SBWComplex[][] T8 = SBW_CLAPACK.Zinverse(T2.data);  // Compute ( - Jac)^-1
////                    //for (int i1 = 0; i1 < Inv.nRows; i1++)
////                    //    for (int j1 = 0; j1 < Inv.nCols; j1++)
////                    //    {
////                    //        Inv[i1, j1].Real = T8[i1][j1].Real;
////                    //        Inv[i1, j1].Imag = T8[i1][j1].Imag;
////                    //    }
////
////
////                    Complex[][] T8 = LA.GetInverse(ConvertComplex(T2.data));
////                    for (int i1 = 0; i1 < Inv.nRows; i1++)
////                        for (int j1 = 0; j1 < Inv.nCols; j1++)
////                        {
////                            Inv[i1, j1].Real = T8[i1][j1].Real;
////                            Inv[i1, j1].Imag = T8[i1][j1].Imag;
////                        }
////
////                    T3.mult(Inv, Nr); // Compute ( - Jac)^-1 . Nr
////
////                    // Finally include the dependent set as well.
////                    T4.mult(LinkMatrix, T3); // Compute L (iwI - Jac)^-1 . Nr
////                    return Matrix.convertToDouble(T4);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException(
////                    "Unexpected error from getUnscaledConcentrationControlCoefficientMatrix()", e.Message);
////            }
////        }
////
////        internal static Complex[][] ConvertComplex(SimpleComplex[][] oMatrix)
////        {
////            var oResult = new Complex[oMatrix.Length][];
////            for (int i = 0; i < oMatrix.Length; i++)
////            {
////                oResult[i] = new Complex[oMatrix[i].Length];
////                for (int j = 0; j < oMatrix[i].Length; j++)
////                {
////                    oResult[i][j] = new Complex();
////                    oResult[i][j].Real = oMatrix[i][j].Real;
////                    oResult[i][j].Imag = oMatrix[i][j].Imag;
////                }
////            }
////            return oResult;
////        }
////
////        [Help("Compute the matrix of scaled concentration control coefficients")]
////        public double[][] getScaledConcentrationControlCoefficientMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] ucc = getUnscaledConcentrationControlCoefficientMatrix();
////
////                    if (ucc.Length > 0)
////                    {
////                        model.convertToConcentrations();
////                        model.computeReactionRates(model.time, model.y);
////                        for (int i = 0; i < ucc.Length; i++)
////                            for (int j = 0; j < ucc[0].Length; j++)
////                            {
////                                ucc[i][j] = ucc[i][j]*model.rates[j]/model.getConcentration(i);
////                            }
////                    }
////                    return ucc;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException(
////                    "Unexpected error from getScaledConcentrationControlCoefficientMatrix()", e.Message);
////            }
////        }
////
////
////        // Use the formula: ucc = elast CS + I
////        [Help("Compute the matrix of unscaled flux control coefficients")]
////        public double[][] getUnscaledFluxControlCoefficientMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] ucc = getUnscaledConcentrationControlCoefficientMatrix();
////                    double[][] uee = getUnscaledElasticityMatrix();
////                    var ucc_m = new Matrix(ucc);
////                    var uee_m = new Matrix(uee);
////
////                    var T1 = new Matrix(uee_m.nRows, ucc_m.nCols);
////                    T1.mult(uee_m, ucc_m);
////                    Matrix T2 = Matrix.Identity(uee.Length);
////                    T1.add(T2);
////                    return Matrix.convertToDouble(T1);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getUnscaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help("Compute the matrix of scaled flux control coefficients")]
////        public double[][] getScaledFluxControlCoefficientMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] ufcc = getUnscaledFluxControlCoefficientMatrix();
////
////                    if (ufcc.Length > 0)
////                    {
////                        model.convertToConcentrations();
////                        model.computeReactionRates(model.time, model.y);
////                        for (int i = 0; i < ufcc.Length; i++)
////                            for (int j = 0; j < ufcc[0].Length; j++)
////                            {
////                                ufcc[i][j] = ufcc[i][j]*model.rates[i]/model.rates[j];
////                            }
////                    }
////                    return ufcc;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        // ----------------------------------------------------------------------------------------------
////
////
////        [Help(
////            "Compute the value for a particular unscaled concentration control coefficients with respect to a local parameter"
////            )]
////        public double getUnscaledConcentrationControlCoefficient(string speciesName, string localReactionName,
////                                                                 string parameterName)
////        {
////            int parameterIndex;
////            int reactionIndex;
////            int speciesIndex;
////            double f1;
////            double f2;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.reactionList.find(localReactionName, out reactionIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate reaction name while computing unscaled control coefficient");
////
////                    if (!ModelGenerator.Instance.floatingSpeciesConcentrationList.find(speciesName, out speciesIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate species name while computing unscaled control coefficient");
////
////                    // Look for the parameter name
////                    if (ModelGenerator.Instance.localParameterList[reactionIndex].find(parameterName,
////                                                                                       out parameterIndex))
////                    {
////                        double originalParameterValue = model.lp[reactionIndex][parameterIndex];
////                        double hstep = DiffStepSize*originalParameterValue;
////                        if (Math.Abs(hstep) < 1E-12)
////                            hstep = DiffStepSize;
////
////                        try
////                        {
////                            model.convertToConcentrations();
////                            model.lp[reactionIndex][parameterIndex] = originalParameterValue + hstep;
////                            model.computeReactionRates(model.time, model.y);
////                            double fi = model.getConcentration(speciesIndex);
////
////                            model.lp[reactionIndex][parameterIndex] = originalParameterValue + 2*hstep;
////                            model.computeReactionRates(model.time, model.y);
////                            double fi2 = model.getConcentration(speciesIndex);
////
////                            model.lp[reactionIndex][parameterIndex] = originalParameterValue - hstep;
////                            model.computeReactionRates(model.time, model.y);
////                            double fd = model.getConcentration(speciesIndex);
////
////                            model.lp[reactionIndex][parameterIndex] = originalParameterValue - 2*hstep;
////                            model.computeReactionRates(model.time, model.y);
////                            double fd2 = model.getConcentration(speciesIndex);
////
////                            // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                            // The following separated lines avoid small amounts of roundoff error
////                            f1 = fd2 + 8*fi;
////                            f2 = -(8*fd + fi2);
////                        }
////                        finally
////                        {
////                            // What ever happens, make sure we restore the species level
////                            model.lp[reactionIndex][parameterIndex] = originalParameterValue;
////                        }
////                        return 1/(12*hstep)*(f1 + f2);
////                    }
////                    else
////                        throw new SBWApplicationException("Unable to locate local parameter [" + parameterName +
////                                                          "] in reaction [" + localReactionName + "]");
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help(
////            "Compute the value for a particular scaled concentration control coefficients with respect to a local parameter"
////            )]
////        public double getScaledConcentrationControlCoefficient(string speciesName, string localReactionName,
////                                                               string parameterName)
////        {
////            int localReactionIndex;
////            int parameterIndex;
////            int speciesIndex;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    double ucc = getUnscaledConcentrationControlCoefficient(speciesName, localReactionName,
////                                                                            parameterName);
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    ModelGenerator.Instance.reactionList.find(localReactionName, out localReactionIndex);
////                    ModelGenerator.Instance.floatingSpeciesConcentrationList.find(localReactionName, out speciesIndex);
////                    ModelGenerator.Instance.localParameterList[localReactionIndex].find(parameterName,
////                                                                                        out parameterIndex);
////
////                    return ucc*model.lp[localReactionIndex][parameterIndex]/model.getConcentration(speciesIndex);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help(
////            "Compute the value for a particular concentration control coefficient, permitted parameters include global parameters, boundary conditions and conservation totals"
////            )]
////        public double getUnscaledConcentrationControlCoefficient(string speciesName, string parameterName)
////        {
////            int speciesIndex;
////            int parameterIndex;
////            TParameterType parameterType;
////            double originalParameterValue;
////            double f1;
////            double f2;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.floatingSpeciesConcentrationList.find(speciesName, out speciesIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate species name while computing unscaled control coefficient");
////
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptGlobalParameter;
////                        originalParameterValue = model.gp[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptBoundaryParameter;
////                        originalParameterValue = model.bc[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptConservationParameter;
////                        originalParameterValue = model.ct[parameterIndex];
////                    }
////                    else throw new SBWApplicationException("Unable to locate parameter: [" + parameterName + "]");
////
////                    double hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
////                        steadyState();
////                        model.convertToConcentrations();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi = model.getConcentration(speciesIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi2 = model.getConcentration(speciesIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd = model.getConcentration(speciesIndex);
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd2 = model.getConcentration(speciesIndex);
////
////                        // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        // What ever happens, make sure we restore the species level
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue);
////                        steadyState();
////                    }
////                    return 1/(12*hstep)*(f1 + f2);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help(
////            "Compute the value for a particular scaled concentration control coefficients with respect to a global or boundary species parameter"
////            )]
////        public double getScaledConcentrationControlCoefficient(string speciesName, string parameterName)
////        {
////            int parameterIndex;
////            int speciesIndex;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    double ucc = getUnscaledConcentrationControlCoefficient(speciesName, parameterName);
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    ModelGenerator.Instance.floatingSpeciesConcentrationList.find(speciesName, out speciesIndex);
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                        return ucc*model.gp[parameterIndex]/model.getConcentration(speciesIndex);
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                        return ucc*model.bc[parameterIndex]/model.getConcentration(speciesIndex);
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                        return ucc*model.ct[parameterIndex]/model.getConcentration(speciesIndex);
////                    return 0.0;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        // ----------------------------------------------------------------------------------------------
////
////
////        [Help("Compute the value for a particular unscaled flux control coefficients with respect to a local parameter")
////        ]
////        public double getUnscaledFluxControlCoefficient(string fluxName, string localReactionName, string parameterName)
////        {
////            int parameterIndex;
////            int localReactionIndex;
////            int fluxIndex;
////            double f1;
////            double f2;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.reactionList.find(localReactionName, out localReactionIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate reaction name while computing unscaled control coefficient");
////
////                    if (!ModelGenerator.Instance.reactionList.find(fluxName, out fluxIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate reaction name while computing unscaled control coefficient");
////
////                    // Look for the parameter name
////                    if (ModelGenerator.Instance.localParameterList[localReactionIndex].find(parameterName,
////                                                                                            out parameterIndex))
////                    {
////                        double originalParameterValue = model.lp[localReactionIndex][parameterIndex];
////                        double hstep = DiffStepSize*originalParameterValue;
////                        if (Math.Abs(hstep) < 1E-12)
////                            hstep = DiffStepSize;
////
////                        try
////                        {
////                            model.convertToConcentrations();
////                            model.lp[localReactionIndex][parameterIndex] = originalParameterValue + hstep;
////                            steadyState();
////                            model.computeReactionRates(model.time, model.y);
////                            double fi = model.rates[fluxIndex];
////
////                            model.lp[localReactionIndex][parameterIndex] = originalParameterValue + 2*hstep;
////                            steadyState();
////                            model.computeReactionRates(model.time, model.y);
////                            double fi2 = model.rates[fluxIndex];
////
////                            model.lp[localReactionIndex][parameterIndex] = originalParameterValue - hstep;
////                            steadyState();
////                            model.computeReactionRates(model.time, model.y);
////                            double fd = model.rates[fluxIndex];
////
////                            model.lp[localReactionIndex][parameterIndex] = originalParameterValue - 2*hstep;
////                            steadyState();
////                            model.computeReactionRates(model.time, model.y);
////                            double fd2 = model.rates[fluxIndex];
////
////                            // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                            // The following separated lines avoid small amounts of roundoff error
////                            f1 = fd2 + 8*fi;
////                            f2 = -(8*fd + fi2);
////                        }
////                        finally
////                        {
////                            // What ever happens, make sure we restore the species level
////                            model.lp[localReactionIndex][parameterIndex] = originalParameterValue;
////                            steadyState();
////                        }
////                        return 1/(12*hstep)*(f1 + f2);
////                    }
////                    else
////                        throw new SBWApplicationException("Unable to locate local parameter [" + parameterName +
////                                                          "] in reaction [" + localReactionName + "]");
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help(
////            "Compute the value for a particular flux control coefficient, permitted parameters include global parameters, boundary conditions and conservation totals"
////            )]
////        public double getUnscaledFluxControlCoefficient(string reactionName, string parameterName)
////        {
////            int fluxIndex;
////            int parameterIndex;
////            TParameterType parameterType;
////            double originalParameterValue;
////            double f1;
////            double f2;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    if (!ModelGenerator.Instance.reactionList.find(reactionName, out fluxIndex))
////                        throw new SBWApplicationException(
////                            "Internal Error: unable to locate species name while computing unscaled control coefficient");
////
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptGlobalParameter;
////                        originalParameterValue = model.gp[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptBoundaryParameter;
////                        originalParameterValue = model.bc[parameterIndex];
////                    }
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                    {
////                        parameterType = TParameterType.ptConservationParameter;
////                        originalParameterValue = model.ct[parameterIndex];
////                    }
////                    else throw new SBWApplicationException("Unable to locate parameter: [" + parameterName + "]");
////
////                    double hstep = DiffStepSize*originalParameterValue;
////                    if (Math.Abs(hstep) < 1E-12)
////                        hstep = DiffStepSize;
////
////                    try
////                    {
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi = model.rates[fluxIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue + 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fi2 = model.rates[fluxIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd = model.rates[fluxIndex];
////
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue - 2*hstep);
////                        steadyState();
////                        model.computeReactionRates(model.time, model.y);
////                        double fd2 = model.rates[fluxIndex];
////
////                        // Use instead the 5th order approximation double unscaledElasticity = (0.5/hstep)*(fi-fd);
////                        // The following separated lines avoid small amounts of roundoff error
////                        f1 = fd2 + 8*fi;
////                        f2 = -(8*fd + fi2);
////                    }
////                    finally
////                    {
////                        // What ever happens, make sure we restore the species level
////                        setParameterValue(parameterType, parameterIndex, originalParameterValue);
////                        steadyState();
////                    }
////                    return 1/(12*hstep)*(f1 + f2);
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help("Compute the value for a particular scaled flux control coefficients with respect to a local parameter")]
////        public double getScaledFluxControlCoefficient(string reactionName, string localReactionName,
////                                                      string parameterName)
////        {
////            int parameterIndex;
////            int reactionIndex;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    double ufcc = getUnscaledFluxControlCoefficient(reactionName, localReactionName, parameterName);
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex);
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                        return ufcc*model.gp[parameterIndex]/model.rates[reactionIndex];
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                        return ufcc*model.bc[parameterIndex]/model.rates[reactionIndex];
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                        return ufcc*model.ct[parameterIndex]/model.rates[reactionIndex];
////                    return 0.0;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////
////
////        [Help(
////            "Compute the value for a particular scaled flux control coefficients with respect to a global or boundary species parameter"
////            )]
////        public double getScaledFluxControlCoefficient(string reactionName, string parameterName)
////        {
////            int parameterIndex;
////            int reactionIndex;
////
////            try
////            {
////                if (modelLoaded)
////                {
////                    double ufcc = getUnscaledFluxControlCoefficient(reactionName, parameterName);
////
////                    model.convertToConcentrations();
////                    model.computeReactionRates(model.time, model.y);
////
////                    ModelGenerator.Instance.reactionList.find(reactionName, out reactionIndex);
////                    if (ModelGenerator.Instance.globalParameterList.find(parameterName, out parameterIndex))
////                        return ufcc*model.gp[parameterIndex]/model.rates[reactionIndex];
////                    else if (ModelGenerator.Instance.boundarySpeciesList.find(parameterName, out parameterIndex))
////                        return ufcc*model.bc[parameterIndex]/model.rates[reactionIndex];
////                    else if (ModelGenerator.Instance.conservationList.find(parameterName, out parameterIndex))
////                        return ufcc*model.ct[parameterIndex]/model.rates[reactionIndex];
////                    return 0.0;
////                }
////                else throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getScaledFluxControlCoefficientMatrix()",
////                                                  e.Message);
////            }
////        }
////    }
////}

//-------------------------------------------------------------------------------------------------------

////using System;
////using System.Collections;
////using System.Collections.Generic;
////using System.Diagnostics;
////using System.IO;
////using System.Threading;
////using LibRoadRunner.Solvers;
////using LibRoadRunner.Util;
////using LibRoadRunner.Util.Unused;
////using libstructural;
////using SBMLSupport;
////using SBW;
////
////namespace LibRoadRunner
////{
////    /// <summary>
////    /// Summary description for RoadRunner.
////    /// </summary>
////    public partial class RoadRunner
////    {
////        #region TSelectionType enum
////
////        public enum TSelectionType
////        {
////            clTime,
////            clBoundarySpecies,
////            clFloatingSpecies,
////            clFlux,
////            clRateOfChange,
////            clVolume,
////            clParameter,
////            clFloatingAmount,
////            clBoundaryAmount,
////            clElasticity,
////            clUnscaledElasticity,
////            clEigenValue,
////            clUnknown,
////            clStoichiometry
////        } ;
////
////        #endregion
////
////        private const double DiffStepSize = 0.05;
////        private const string emptyModelStr = "A model needs to be loaded before one can use this method";
////        private const double STEADYSTATE_THRESHOLD = 1E-2;
////
////        public static bool _bComputeAndAssignConservationLaws = true;
////        public static bool _bConservedTotalChanged;
////        public static bool _ReMultiplyCompartments;
////        public double[][] _L;
////        public double[][] _L0;
////        public double[][] _N;
////        public double[][] _Nr;
////        //private ArrayList _oSteadyStateSelection;
////        private TSelectionRecord[] _oSteadyStateSelection;
////        private string _sModelCode;
////
////        private CvodeInterface cvode;
////        //kinSolverInterface kinSolver;  // Use NLEQ1 instead
////
////        public IModel model = null;
////        public bool modelLoaded = false;
////        private ISteadyStateSolver steadyStateSolver;
////        public int numPoints;
////        public string sbmlStr;
////        private TSelectionRecord[] selectionList;
////        public double timeEnd;
////        public double timeStart;
////
////        public RoadRunner()
////        {
////
////            System.Globalization.CultureInfo culture = System.Globalization.CultureInfo.CreateSpecificCulture("en");
////            culture.NumberFormat.NumberDecimalSeparator = ".";
////            Thread.CurrentThread.CurrentCulture = culture;
////            // Set up some defaults
////            timeStart = 0;
////            timeEnd = 10;
////            numPoints = 21;
////            sbmlStr = "";
////
////            UseKinsol = IsNleqAvailable() ? 0 : 1;
////        }
////
////        private bool IsNleqAvailable()
////        {
////            return NLEQInterface.IsAvailable;
////        }
////#if DEBUG
////        public static void Test()
////        {
////
////            double[,] results;
////            RoadRunner sim;
////            sim = new RoadRunner();
////            //RoadRunner.ReMultiplyCompartments(false);
////            //RoadRunner.ComputeAndAssignConservationLaws(false);
////            sim.setTolerances(1E-4, 1E-4, 100);
////            //sim.loadSBMLFromFile(@"C:\Development\sbwBuild\source\Translators\TestModels\MathMLTests.xml");
////            //sim.loadSBMLFromFile(@"C:\Development\trunk-sbml\trunk\test-suite\cases\semantic\00938\00938-sbml-l3v1.xml");
////            //sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00978\00978-sbml-l3v1.xml");
////            string test = "01104";
////            sim.loadSBMLFromFile(string.Format(@"C:\Development\test-suite\cases\semantic\{0}\{0}-sbml-l3v1.xml", test));
////
////            //sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00951\00951-sbml-l3v1.xml");
////            //sim.setSelectionList(new ArrayList(new string[] {
////            //    "time", "x", "y", "p", "q"
////            //    }));
////            //results = sim.simulateEx(0, 2, 11);
////            ////var writer = new StringWriter();
////
////            //sim.loadSBMLFromFile(@"C:\Users\fbergmann\Desktop\max.xml");
////            sim.setSelectionList(new ArrayList(new string[] {
////                "time", "Xref"
////            }));
////            results = sim.simulateEx(0, 10, 11);
////
////            DumpResults(Console.Out, results, sim.getSelectionList());
////
////
////
////            //double[,] results;
////            //RoadRunner sim;
////            //sim = new RoadRunner();
////            ////sim.loadSBMLFromFile(@"C:\Development\sbwBuild\source\Translators\TestModels\MathMLTests.xml");
////            ////sim.loadSBMLFromFile(@"C:\Development\trunk-sbml\trunk\test-suite\cases\semantic\00938\00938-sbml-l3v1.xml");
////            //sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00952\00952-sbml-l3v1.xml");
////            ////sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00951\00951-sbml-l3v1.xml");
////            //sim.setSelectionList(new ArrayList(new string[] {
////            //    "time", "S", "Q", "R", "reset"
////            //    }));
////            //results = sim.simulateEx(0, 1, 11);
////            ////var writer = new StringWriter();
////            //DumpResults(Console.Out, results, sim.getSelectionList());
////
////            //sim = new RoadRunner();
////            //sim.setTolerances(1e-10, 1e-9);
////            //sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00374\00374-sbml-l2v4.xml");
////            //sim.setSelectionList(new ArrayList(new string[] {
////            //    "time", "S1", "S2", "S3", "S4"
////            //    }));
////            //results = sim.simulateEx(0, 2, 51);
////            //DumpResults(Console.Out, results,sim.getSelectionList());
////
////            //sim = new RoadRunner();
////            //_bComputeAndAssignConservationLaws = false;
////            //sim.loadSBMLFromFile(@"C:\Development\test-suite\cases\semantic\00424\00424-sbml-l3v1.xml");
////            //sim.setSelectionList(new ArrayList(new string[] {
////            //    "time", "S1", "S2", "S3"
////            //    }));
////
////            ////sim.CorrectMaxStep();
////            //CvodeInterface.MaxStep = 0.0001;
////            //sim.cvode.reStart(0.0, sim.model);
////
////            ////sim.cvode.
////            //results = sim.simulateEx(0, 5, 51);
////            //DumpResults(Console.Out, results, sim.getSelectionList());
////
////            //Debug.WriteLine(writer.GetStringBuilder().ToString());
////            //Debug.WriteLine(sim.getWarnings());
////            //Debug.WriteLine(sim.getCapabilities());
////        }
////#endif
////
////        [Ignore]
////        public string NL
////        {
////            [Ignore]
////            get { return _NL(); }
////        }
////
////
////        [Ignore]
////        private string _NL()
////        {
////            return Environment.NewLine;
////        }
////
////
////        [Ignore]
////        private void emptyModel()
////        {
////            throw new SBWApplicationException(emptyModelStr);
////        }
////
////
////        private double GetValueForRecord(TSelectionRecord record)
////        {
////            double dResult;
////            switch (record.selectionType)
////            {
////                case TSelectionType.clFloatingSpecies:
////                    dResult = model.getConcentration(record.index);
////                    break;
////                case TSelectionType.clBoundarySpecies:
////                    dResult = model.bc[record.index];
////                    break;
////                case TSelectionType.clFlux:
////                    dResult = model.rates[record.index];
////                    break;
////                case TSelectionType.clRateOfChange:
////                    dResult = model.dydt[record.index];
////                    break;
////                case TSelectionType.clVolume:
////                    dResult = model.c[record.index];
////                    break;
////                case TSelectionType.clParameter:
////                    {
////                        if (record.index > model.gp.Length - 1)
////                            dResult = model.ct[record.index - model.gp.Length];
////                        else
////                            dResult = model.gp[record.index];
////                    }
////                    break;
////                case TSelectionType.clFloatingAmount:
////                    dResult = model.amounts[record.index];
////                    break;
////                case TSelectionType.clBoundaryAmount:
////                    int nIndex;
////                    if (
////                        ModelGenerator.Instance.compartmentList.find(
////                            ModelGenerator.Instance.boundarySpeciesList[record.index].compartmentName,
////                            out nIndex))
////                        dResult = model.bc[record.index] * model.c[nIndex];
////                    else
////                        dResult = 0.0;
////                    break;
////                case TSelectionType.clElasticity:
////                    dResult = getEE(record.p1, record.p2, false);
////                    break;
////                case TSelectionType.clUnscaledElasticity:
////                    dResult = getuEE(record.p1, record.p2, false);
////                    break;
////                case TSelectionType.clEigenValue:
////                    Complex[] oComplex = LA.GetEigenValues(getReducedJacobian());
////                    if (oComplex.Length > record.index)
////                    {
////                        dResult = oComplex[record.index].Real;
////                    }
////                    else
////                        dResult = Double.NaN;
////                    break;
////                case TSelectionType.clStoichiometry:
////                    dResult = model.sr[record.index];
////                    break;
////                default:
////                    dResult = 0.0;
////                    break;
////            }
////            return dResult;
////        }
////
////        private double GetNthSelectedOutput(int index, double dCurrentTime)
////        {
////            TSelectionRecord record = selectionList[index];
////            if (record.selectionType == TSelectionType.clTime)
////                return dCurrentTime;
////
////            return GetValueForRecord(record);
////        }
////
////        private void AddNthOutputToResult(double[,] results, int nRow, double dCurrentTime)
////        {
////            for (int j = 0; j < selectionList.Length; j++)
////            {
////                results[nRow, j] = GetNthSelectedOutput(j, dCurrentTime);
////            }
////        }
////
////        private double[] BuildModelEvalArgument()
////        {
////            var dResult = new double[model.amounts.Length + model.rateRules.Length];
////            double[] dCurrentRuleValues = model.GetCurrentValues();
////            dCurrentRuleValues.CopyTo(dResult, 0);
////            model.amounts.CopyTo(dResult, model.rateRules.Length);
////            return dResult;
////        }
////
////        [Ignore]
////        public double[,] runSimulation()
////        {
////            double hstep = (timeEnd - timeStart) / (numPoints - 1);
////            var results = new double[numPoints, selectionList.Length];
////
////            model.evalModel(timeStart, BuildModelEvalArgument());
////
////            AddNthOutputToResult(results, 0, timeStart);
////
////            if (cvode.HaveVariables)
////            {
////
////                int restartResult = cvode.reStart(timeStart, model);
////                if (restartResult != 0)
////                    throw new SBWApplicationException("Error in reStart call to CVODE");
////            }
////            double tout = timeStart;
////            for (int i = 1; i < numPoints; i++)
////            {
////                cvode.OneStep(tout, hstep);
////                tout = timeStart + i * hstep;
////                AddNthOutputToResult(results, i, tout);
////            }
////            return results;
////        }
////
////        // -------------------------------------------------------------------------------
////
////#if DEBUG
////        public static void PrintTout(double start, double end, int numPoints)
////        {
////            double hstep = (end - start) / (numPoints - 1);
////            Debug.WriteLine("Using step " + hstep);
////            double tout = start;
////            for (int i = 1; i < numPoints; i++)
////            {
////                tout = start + i*hstep;
////                Debug.WriteLine(tout.ToString("G17"));
////            }
////        }
////#endif
////
////        private void InitializeModel(object o)
////        {
////            model = ((IModel)o);
////            model.Warnings.AddRange(ModelGenerator.Instance.Warnings);
////            modelLoaded = true;
////            _bConservedTotalChanged = false;
////
////            model.setCompartmentVolumes();
////            model.initializeInitialConditions();
////            model.setParameterValues();
////            model.setCompartmentVolumes();
////            model.setBoundaryConditions();
////
////            model.setInitialConditions();
////            model.convertToAmounts();
////
////            model.evalInitialAssignments();
////            model.computeRules(model.y);
////            model.convertToAmounts();
////
////            if (_bComputeAndAssignConservationLaws) model.computeConservedTotals();
////
////            cvode = new CvodeInterface(model);
////
////            reset();
////
////            // Construct default selection list
////            selectionList = new TSelectionRecord[model.getNumTotalVariables + 1]; // + 1 to include time
////            selectionList[0].selectionType = TSelectionType.clTime;
////            for (int i = 0; i < model.getNumTotalVariables; i++)
////            {
////                selectionList[i + 1].index = i;
////                selectionList[i + 1].selectionType = TSelectionType.clFloatingSpecies;
////            }
////
////            _oSteadyStateSelection = null;
////        }
////
////        private static void DumpResults(TextWriter writer, double[,] data, ArrayList colLabels)
////        {
////            for (int i = 0; i < colLabels.Count; i++)
////            {
////                writer.Write(colLabels[i] + "\t");
////                Debug.Write(colLabels[i] + "\t");
////            }
////            writer.WriteLine();
////            Debug.WriteLine("");
////
////            for (int i = 0; i < data.GetLength(0); i++)
////            {
////                for (int j = 0; j < data.GetLength(1); j++)
////                {
////                    writer.Write(data[i, j] + "\t");
////                    Debug.Write(data[i, j] + "\t");
////                }
////                writer.WriteLine();
////                Debug.WriteLine("");
////            }
////        }
////
////
////        //public static void TestDirectory(string directory, bool testSubDirs)
////        //{
////        //    //TestDirectory(directory, testSubDirs, "*sbml-l3v1.xml");
////        //    TestDirectory(directory, testSubDirs, "*.xml");
////        //}
////
////    //    public static void TestDirectory(string directory, bool testSubDirs, string pattern)
////    //{
////    //        var files = Directory.GetFiles(directory, pattern,
////    //            (testSubDirs ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly));
////    //        foreach (var item in files)
////    //        {
////    //            try
////    //            {
////    //                var rr = new RoadRunner();
////    //                rr.setTolerances(1e-6, 1e-3);
////    //                rr.loadSBMLFromFile(item);
////    //                rr.simulateEx(0, 10, 1000);
////    //                Debug.WriteLine(string.Format("File: {0} passed", Path.GetFileName(item)));
////    //            }
////    //            catch (Exception ex)
////    //            {
////    //                Debug.WriteLine(string.Format("File: {0} failed: ", Path.GetFileName(item), ex.Message));
////    //            }
////    //        }
////    //    }
////
////
////        [Ignore]
////        public static void SimulateSBMLFile(string fileName, bool useConservationLaws)
////        {
////            var sim = new RoadRunner();
////            ComputeAndAssignConservationLaws(useConservationLaws);
////            sim.loadSBML(File.ReadAllText(fileName));
////
////            double[,] data = sim.simulate();
////            ArrayList list = sim.getSelectionList();
////            TextWriter writer = Console.Out;
////
////            DumpResults(writer, data, list);
////            return;
////        }
////
////        [Ignore]
////        public static void SimulateSBMLFile(string fileName, bool useConservationLaws, double startTime, double endTime,
////                                            int numPoints)
////        {
////            var sim = new RoadRunner();
////            ComputeAndAssignConservationLaws(useConservationLaws);
////            sim.loadSBML(File.ReadAllText(fileName));
////
////            try
////            {
////                double[,] data = sim.simulateEx(startTime, endTime, numPoints);
////                ArrayList list = sim.getSelectionList();
////                TextWriter writer = Console.Error;
////
////                DumpResults(writer, data, list);
////            }
////            catch (Exception ex)
////            {
////                Debug.WriteLine(ex);
////            }
////
////            //Debug.WriteLine(sim.getCapabilities());
////
////            return;
////        }
////
////        [Help("Load SBML into simulator")]
////        public void loadSBMLFromFile(string fileName)
////        {
////            loadSBML(File.ReadAllText(fileName));
////        }
////
////        [Help("Load SBML into simulator")]
////        public void loadSBML(string sbml)
////        {
////            if (string.IsNullOrEmpty(sbml))
////                throw new SBWApplicationException("Invalid SBML");
////
////
////            // If the user loads the same model again, don't both loading into NOM,
////            // just reset the initial conditions
////
////            if (modelLoaded && model != null && (sbml == sbmlStr) && (sbml != ""))
////            {
////                InitializeModel(model);
////                //reset();
////            }
////            else
////            {
////                if (model != null)
////                {
////                    cvode = null;
////                    model = null;
////                    modelLoaded = false;
////                    GC.Collect();
////                }
////
////                sbmlStr = sbml;
////
////
////                _sModelCode = ModelGenerator.Instance.generateModelCode(sbmlStr);
////
////                string sLocation = GetType().Assembly.Location;
////                Compiler.addAssembly(typeof(MathKGI).Assembly.Location);
////                Compiler.addAssembly(typeof(System.Collections.Generic.List<string>).Assembly.Location);
////
////                object o = Compiler.getInstance(_sModelCode, "TModel", sLocation);
////
////                if (o != null)
////                {
////                    InitializeModel(o);
////                }
////                else
////                {
////                    model = null;
////                    modelLoaded = false;
////                    try
////                    {
////                        var sw = new StreamWriter(Environment.GetEnvironmentVariable("TEMP") + "/SBW_ErrorLog.txt");
////                        try
////                        {
////                            sw.WriteLine("ErrorMessage: ");
////                            sw.WriteLine(Compiler.getLastErrors());
////                            sw.WriteLine("C# Model Code: ");
////                            sw.Write(_sModelCode);
////                        }
////                        finally
////                        {
////                            sw.Close();
////                        }
////                    }
////                    catch (Exception)
////                    {
////                    }
////                    throw new SBWApplicationException("Internal Error: The model has failed to compile." + NL
////                                                      + "The model file has been deposited at " +
////                                                      Environment.GetEnvironmentVariable("TEMP") + "/SBW_ErrorLog.txt",
////                                                      Compiler.getLastErrors());
////                }
////
////                _L = StructAnalysis.GetLinkMatrix();
////                _L0 = StructAnalysis.GetL0Matrix();
////                _N = StructAnalysis.GetReorderedStoichiometryMatrix();
////                _Nr = StructAnalysis.GetNrMatrix();
////            }
////        }
////
////
////        [Help("Returns the initially loaded model as SBML")]
////        public string getSBML()
////        {
////            return sbmlStr;
////        }
////
////        [Help("get the currently set time start")]
////        public double getTimeStart()
////        {
////            return timeStart;
////        }
////
////        [Help("get the currently set time end")]
////        public double getTimeEnd()
////        {
////            return timeEnd;
////        }
////
////        [Help("get the currently set number of points")]
////        public int getNumPoints()
////        {
////            return numPoints;
////        }
////
////        [Help("Set the time start for the simulation")]
////        public void setTimeStart(double startTime)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if (startTime < 0)
////                throw new SBWApplicationException("Time Start most be greater than zero");
////            this.timeStart = startTime;
////        }
////
////        [Help("Set the time end for the simulation")]
////        public void setTimeEnd(double endTime)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if (endTime <= 0)
////                throw new SBWApplicationException("Time End most be greater than zero");
////            this.timeEnd = endTime;
////        }
////
////        [Help("Set the number of points to generate during the simulation")]
////        public void setNumPoints(int nummberOfPoints)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if (nummberOfPoints <= 0)
////                nummberOfPoints = 1;
////            this.numPoints = nummberOfPoints;
////        }
////
////        [Help("reset the simulator back to the initial conditions specified in the SBML model")]
////        public void reset()
////        {
////            if (!modelLoaded)
////            {
////                // rather make sure that the simulator is!!!! in a stable state
////                model = null;
////                sbmlStr = "";
////            }
////            else
////            {
////                model.time = 0.0;
////
////                // Reset the event flags
////                model.resetEvents();
////
////                model.setCompartmentVolumes();
////
////                model.setInitialConditions();
////
////                model.convertToAmounts();
////
////                // in case we have ODE rules we should assign those as initial values
////                model.InitializeRateRuleSymbols();
////                model.InitializeRates();
////                // and of course initial assignments should override anything
////                model.evalInitialAssignments();
////                model.convertToAmounts();
////                // also we might need to set some initial assignment rules.
////                model.convertToConcentrations();
////                model.computeRules(model.y);
////                model.InitializeRates();
////                model.InitializeRateRuleSymbols();
////                model.evalInitialAssignments();
////                model.computeRules(model.y);
////
////                model.convertToAmounts();
////
////                if (_bComputeAndAssignConservationLaws && !_bConservedTotalChanged) model.computeConservedTotals();
////
////                cvode.AssignNewVector(model, true);
////                cvode.TestRootsAtInitialTime();
////
////                //double hstep = (timeEnd - timeStart) / (numPoints - 1);
////                //CvodeInterface.MaxStep = Math.Min(CvodeInterface.MaxStep, hstep);
////                //if (CvodeInterface.MaxStep == 0)
////                //    CvodeInterface.MaxStep = hstep;
////
////
////                model.time = 0.0;
////                cvode.reStart(0.0, model);
////
////                cvode.assignments.Clear();
////
////                try
////                {
////                    model.testConstraints();
////                }
////                catch (Exception e)
////                {
////                    model.Warnings.Add("Constraint Violated at time = 0\n" + e.Message);
////                }
////            }
////        }
////
////        [Help(
////            "Change the initial conditions to another concentration vector (changes only initial conditions for floating Species)"
////            )]
////        public void changeInitialConditions(double[] ic)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            for (int i = 0; i < ic.Length; i++)
////            {
////                model.setConcentration(i, ic[i]);
////                if (model.init_y.Length > i)
////                model.init_y[i] = ic[i];
////            }
////            model.convertToAmounts();
////            model.computeConservedTotals();
////        }
////
////
////        [Help("Carry out a time course simulation")]
////        public double[,] simulate()
////        {
////            try
////            {
////                if (!modelLoaded)
////                    throw new SBWApplicationException(emptyModelStr);
////                if (timeEnd <= timeStart)
////                    throw new SBWApplicationException("Error: time end must be greater than time start");
////                return runSimulation();
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from simulate(): " + e.Message);
////            }
////        }
////
////        [Help(
////            "Extension method to simulate (time start, time end, number of points). This routine resets the model to its initial condition before running the simulation (unlike simulate())"
////            )]
////        public double[,] simulateEx(double startTime, double endTime, int numberOfPoints)
////        {
////            try
////            {
////                if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////                reset(); // reset back to initial conditions
////
////                if (endTime < 0 || startTime < 0 || numberOfPoints <= 0 || endTime <= startTime)
////                    throw new SBWApplicationException("Illegal input to simulateEx");
////
////                this.timeEnd = endTime;
////                this.timeStart = startTime;
////                numPoints = numberOfPoints;
////                return runSimulation();
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from simulateEx()", e.Message);
////            }
////        }
////
////        [Help("Returns the current vector of reactions rates")]
////        public double[] getReactionRates()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            model.convertToConcentrations();
////            model.computeReactionRates(0.0, model.y);
////            return model.rates;
////        }
////
////        [Help("Returns the current vector of rates of change")]
////        public double[] getRatesOfChange()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            model.computeAllRatesOfChange();
////            return model.dydt;
////        }
////
////        [Help(
////            "Returns a list of floating species names: This method is deprecated, please use getFloatingSpeciesNames()")
////        ]
////        public ArrayList getSpeciesNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            return ModelGenerator.Instance.getFloatingSpeciesConcentrationList(); // Reordered list
////        }
////
////        [Help("Returns a list of reaction names")]
////        public ArrayList getReactionNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            return ModelGenerator.Instance.getReactionNames();
////        }
////
////
////        // ---------------------------------------------------------------------
////        // Start of Level 2 API Methods
////        // ---------------------------------------------------------------------
////
////        public int UseKinsol { get; set; }
////
////        [Help("Get Simulator Capabilities")]
////        public string getCapabilities()
////        {
////            CapsSupport current = CapsSupport.CurrentSettings;
////            current["integration"].Capabilities.Add(new CapsSupport.Capability
////            {
////                Name = "usekinsol", IntValue = UseKinsol, Hint = "Is KinSol used as steady state integrator", Type = "int"
////            }
////                );
////
////            return current.ToXml();
////        }
////
////        [Ignore]
////        public void setTolerances(double aTol, double rTol)
////        {
////            CvodeInterface.relTol = rTol;
////            CvodeInterface.absTol = aTol;
////        }
////
////        [Ignore]
////        public void setTolerances(double aTol, double rTol, int maxSteps)
////        {
////            setTolerances(aTol, rTol);
////            CvodeInterface.MaxNumSteps = maxSteps;
////        }
////
////        public void CorrectMaxStep()
////        {
////            double maxStep = (timeEnd - timeStart) / (numPoints);
////            maxStep = Math.Min(CvodeInterface.MaxStep, maxStep);
////            CvodeInterface.MaxStep = maxStep;
////        }
////
////        [Help("Set Simulator Capabilites")]
////        public void setCapabilities(string capsStr)
////        {
////            var cs = new CapsSupport(capsStr);
////            cs.Apply();
////
////            //CorrectMaxStep();
////
////            if (modelLoaded)
////            {
////                cvode = new CvodeInterface(model);
////                for (int i = 0; i < model.getNumIndependentVariables; i++)
////                {
////                    cvode.setAbsTolerance(i, CvodeInterface.absTol);
////                }
////                cvode.reStart(0.0, model);
////            }
////
////            if (cs.HasSection("integration") && cs["integration"].HasCapability("usekinsol"))
////            {
////
////                CapsSupport.Capability cap = cs["integration", "usekinsol"];
////                UseKinsol = cap.IntValue;
////            }
////
////        }
////
////        [Help("Sets the value of the given species or global parameter to the given value (not of local parameters)")]
////        public void setValue(string sId, double dValue)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////
////            int nIndex = -1;
////            if (ModelGenerator.Instance.globalParameterList.find(sId, out nIndex))
////            {
////                model.gp[nIndex] = dValue;
////                return;
////            }
////            if (ModelGenerator.Instance.boundarySpeciesList.find(sId, out nIndex))
////            {
////                model.bc[nIndex] = dValue;
////                return;
////            }
////            if (ModelGenerator.Instance.compartmentList.find(sId, out nIndex))
////            {
////                model.c[nIndex] = dValue;
////                return;
////            }
////            if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(sId, out nIndex))
////            {
////                model.setConcentration(nIndex, dValue);
////                model.convertToAmounts();
////                if (!_bConservedTotalChanged) model.computeConservedTotals();
////                return;
////            }
////            if (ModelGenerator.Instance.conservationList.find(sId, out nIndex))
////            {
////                model.ct[nIndex] = dValue;
////                model.updateDependentSpeciesValues(model.y);
////                _bConservedTotalChanged = true;
////                return;
////            }
////
////            var initialConditions =
////                new List<string>((string[])getFloatingSpeciesInitialConditionNames().ToArray(typeof(string)));
////            if (initialConditions.Contains(sId))
////            {
////                int index = initialConditions.IndexOf(sId);
////                model.init_y[index] = dValue;
////                reset();
////                return;
////            }
////
////
////            throw new SBWApplicationException(String.Format("Given Id: '{0}' not found.", sId),
////                                              "Only species and global parameter values can be set");
////        }
////
////
////        [Help("Gets the Value of the given species or global parameter (not of local parameters)")]
////        public double getValue(string sId)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            int nIndex = 0;
////            if (ModelGenerator.Instance.globalParameterList.find(sId, out nIndex))
////            {
////                return model.gp[nIndex];
////            }
////            if (ModelGenerator.Instance.boundarySpeciesList.find(sId, out nIndex))
////            {
////                return model.bc[nIndex];
////            }
////            if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(sId, out nIndex))
////            {
////                return model.y[nIndex];
////            }
////
////            if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(sId.Substring(0, sId.Length - 1),
////                                                                              out nIndex))
////            {
////                //fs[j] + "'" will be interpreted as rate of change
////                return model.dydt[nIndex];
////            }
////
////            if (ModelGenerator.Instance.compartmentList.find(sId, out nIndex))
////            {
////                return model.c[nIndex];
////            }
////            if (ModelGenerator.Instance.reactionList.find(sId, out nIndex))
////            {
////                return model.rates[nIndex];
////            }
////
////            if (ModelGenerator.Instance.conservationList.find(sId, out nIndex))
////            {
////                return model.ct[nIndex];
////            }
////
////            var initialConditions =
////                new List<string>((string[])getFloatingSpeciesInitialConditionNames().ToArray(typeof(string)));
////            if (initialConditions.Contains(sId))
////            {
////                int index = initialConditions.IndexOf(sId);
////                return model.init_y[index];
////            }
////
////            if (sId.StartsWith("EE:"))
////            {
////                string parameters = sId.Substring(3);
////                var p1 = parameters.Substring(0, parameters.IndexOf(","));
////                var p2 = parameters.Substring(parameters.IndexOf(",") + 1);
////                return getEE(p1, p2, false);
////            }
////
////
////            if (sId.StartsWith("uEE:"))
////            {
////                string parameters = sId.Substring(4);
////                var p1 = parameters.Substring(0, parameters.IndexOf(","));
////                var p2 = parameters.Substring(parameters.IndexOf(",") + 1);
////                return getuEE(p1, p2, false);
////            }
////
////            if (sId.StartsWith("eigen_"))
////            {
////                var species = (sId).Substring("eigen_".Length);
////                int index;
////                ModelGenerator.Instance.floatingSpeciesConcentrationList.find(species, out index);
////                Complex[] oComplex = LA.GetEigenValues(getReducedJacobian());
////                if (oComplex.Length > selectionList[index].index)
////                {
////                    return oComplex[selectionList[index].index].Real;
////                }
////                return Double.NaN;
////            }
////
////            throw new SBWApplicationException("Given Id: '" + sId + "' not found.",
////                                              "Only species, global parameter values and fluxes can be returned");
////        }
////
////        [Help(
////            "Returns symbols of the currently loaded model, that can be used for the selectionlist format array of arrays  { { \"groupname\", { \"item1\", \"item2\" ... } } }."
////            )]
////        public ArrayList getAvailableSymbols()
////        {
////            var oResult = new ArrayList {new ArrayList(new object[] {"Time", new ArrayList(new object[] {"time"})})};
////
////            if (!modelLoaded) return oResult;
////
////            oResult.Add(new ArrayList(new object[] { "Floating Species", getFloatingSpeciesNames() }));
////            oResult.Add(new ArrayList(new object[] { "Boundary Species", getBoundarySpeciesNames() }));
////            oResult.Add(new ArrayList(new object[] { "Floating Species (amount)", getFloatingSpeciesAmountNames() }));
////            oResult.Add(new ArrayList(new object[] { "Boundary Species (amount)", getBoundarySpeciesAmountNames() }));
////            oResult.Add(new ArrayList(new object[] { "Global Parameters", getParameterNames() }));
////            oResult.Add(new ArrayList(new object[] { "Fluxes", getReactionNames() }));
////            oResult.Add(new ArrayList(new object[] { "Rates of Change", getRateOfChangeNames() }));
////            oResult.Add(new ArrayList(new object[] { "Volumes", ModelGenerator.Instance.getCompartmentList() }));
////            oResult.Add(new ArrayList(new object[] { "Elasticity Coefficients", getElasticityCoefficientNames() }));
////            oResult.Add(
////                new ArrayList(new object[] { "Unscaled Elasticity Coefficients", getUnscaledElasticityCoefficientNames() }));
////            oResult.Add(new ArrayList(new object[] { "Eigenvalues", getEigenValueNames() }));
////
////            return oResult;
////        }
////
////
////        [Help("Returns the currently selected columns that will be returned by calls to simulate() or simulateEx(,,).")]
////        public ArrayList getSelectionList()
////        {
////            var oResult = new ArrayList();
////
////            if (!modelLoaded)
////            {
////                oResult.Add("time");
////                return oResult;
////            }
////
////            ArrayList oFloating = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oFluxes = ModelGenerator.Instance.getReactionNames();
////            ArrayList oVolumes = ModelGenerator.Instance.getCompartmentList();
////            ArrayList oRates = getRateOfChangeNames();
////            ArrayList oParameters = getParameterNames();
////
////
////            foreach (TSelectionRecord record in selectionList)
////            {
////                switch (record.selectionType)
////                {
////                    case TSelectionType.clTime:
////                        oResult.Add("time");
////                        break;
////                    case TSelectionType.clBoundaryAmount:
////                        oResult.Add(string.Format("[{0}]", oBoundary[record.index]));
////                        break;
////                    case TSelectionType.clBoundarySpecies:
////                        oResult.Add(oBoundary[record.index]);
////                        break;
////                    case TSelectionType.clFloatingAmount:
////                        oResult.Add(string.Format("[{0}]", oFloating[record.index]));
////                        break;
////                    case TSelectionType.clFloatingSpecies:
////                        oResult.Add(oFloating[record.index]);
////                        break;
////                    case TSelectionType.clVolume:
////                        oResult.Add(oVolumes[record.index]);
////                        break;
////                    case TSelectionType.clFlux:
////                        oResult.Add(oFluxes[record.index]);
////                        break;
////                    case TSelectionType.clRateOfChange:
////                        oResult.Add(oRates[record.index]);
////                        break;
////                    case TSelectionType.clParameter:
////                        oResult.Add(oParameters[record.index]);
////                        break;
////                    case TSelectionType.clEigenValue:
////                        oResult.Add("eigen_" + record.p1);
////                        break;
////                    case TSelectionType.clElasticity:
////                        oResult.Add(String.Format("EE:{0},{1}", record.p1, record.p2));
////                        break;
////                    case TSelectionType.clUnscaledElasticity:
////                        oResult.Add(String.Format("uEE:{0},{1}", record.p1, record.p2));
////                        break;
////                    case TSelectionType.clStoichiometry:
////                        oResult.Add(record.p1);
////                        break;
////                }
////            }
////
////            return oResult;
////        }
////
////
////        [Help("Set the columns to be returned by simulate() or simulateEx(), valid symbol names include" +
////              " time, species names, , volume, reaction rates and rates of change (speciesName')")]
////        public void setSelectionList(ArrayList newSelectionList)
////        {
////            selectionList = new TSelectionRecord[newSelectionList.Count];
////            ArrayList fs = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList bs = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList rs = ModelGenerator.Instance.getReactionNames();
////            ArrayList vol = ModelGenerator.Instance.getCompartmentList();
////            ArrayList gp = ModelGenerator.Instance.getGlobalParameterList();
////            var sr = ModelGenerator.Instance.ModifiableSpeciesReferenceList;
////
////            for (int i = 0; i < newSelectionList.Count; i++)
////            {
////                // Check for species
////                for (int j = 0; j < fs.Count; j++)
////                {
////                    if ((string)newSelectionList[i] == (string)fs[j])
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clFloatingSpecies;
////                        break;
////                    }
////
////                    if ((string)newSelectionList[i] == "[" + (string)fs[j] + "]")
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clFloatingAmount;
////                        break;
////                    }
////
////                    // Check for species rate of change
////                    if ((string)newSelectionList[i] == (string)fs[j] + "'")
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clRateOfChange;
////                        break;
////                    }
////                }
////
////                // Check fgr boundary species
////                for (int j = 0; j < bs.Count; j++)
////                {
////                    if ((string)newSelectionList[i] == (string)bs[j])
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clBoundarySpecies;
////                        break;
////                    }
////                    if ((string)newSelectionList[i] == "[" + (string)bs[j] + "]")
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clBoundaryAmount;
////                        break;
////                    }
////                }
////
////
////                if ((string)newSelectionList[i] == "time")
////                {
////                    selectionList[i].selectionType = TSelectionType.clTime;
////                }
////
////                for (int j = 0; j < rs.Count; j++)
////                {
////                    // Check for reaction rate
////                    if ((string)newSelectionList[i] == (string)rs[j])
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clFlux;
////                        break;
////                    }
////                }
////
////                for (int j = 0; j < vol.Count; j++)
////                {
////                    // Check for volume
////                    if ((string)newSelectionList[i] == (string)vol[j])
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clVolume;
////                        break;
////                    }
////                }
////
////                for (int j = 0; j < gp.Count; j++)
////                {
////                    // Check for volume
////                    if ((string)newSelectionList[i] == (string)gp[j])
////                    {
////                        selectionList[i].index = j;
////                        selectionList[i].selectionType = TSelectionType.clParameter;
////                        break;
////                    }
////                }
////
////                if (((string)newSelectionList[i]).StartsWith("EE:"))
////                {
////                    string parameters = ((string)newSelectionList[i]).Substring(3);
////                    var p1 = parameters.Substring(0, parameters.IndexOf(","));
////                    var p2 = parameters.Substring(parameters.IndexOf(",") + 1);
////                    selectionList[i].selectionType = TSelectionType.clElasticity;
////                    selectionList[i].p1 = p1;
////                    selectionList[i].p2 = p2;
////                }
////
////                if (((string)newSelectionList[i]).StartsWith("uEE:"))
////                {
////                    string parameters = ((string)newSelectionList[i]).Substring(4);
////                    var p1 = parameters.Substring(0, parameters.IndexOf(","));
////                    var p2 = parameters.Substring(parameters.IndexOf(",") + 1);
////                    selectionList[i].selectionType = TSelectionType.clUnscaledElasticity;
////                    selectionList[i].p1 = p1;
////                    selectionList[i].p2 = p2;
////                }
////                if (((string)newSelectionList[i]).StartsWith("eigen_"))
////                {
////                    var species = ((string)newSelectionList[i]).Substring("eigen_".Length);
////                    selectionList[i].selectionType = TSelectionType.clEigenValue;
////                    selectionList[i].p1 = species;
////                    ModelGenerator.Instance.floatingSpeciesConcentrationList.find(species, out selectionList[i].index);
////                }
////
////                int index;
////                if (sr.find((string)newSelectionList[i], out index))
////                {
////                    selectionList[i].selectionType = TSelectionType.clStoichiometry;
////                    selectionList[i].index = index;
////                    selectionList[i].p1 = (string) newSelectionList[i];
////                }
////
////            }
////        }
////
////
////        [Help(
////            "Carry out a single integration step using a stepsize as indicated in the method call (the intergrator is reset to take into account all variable changes). Arguments: double CurrentTime, double StepSize, Return Value: new CurrentTime."
////            )]
////        public double oneStep(double currentTime, double stepSize)
////        {
////            return oneStep(currentTime, stepSize, true);
////        }
////
////        [Help(
////           "Carry out a single integration step using a stepsize as indicated in the method call. Arguments: double CurrentTime, double StepSize, bool: reset integrator if true, Return Value: new CurrentTime."
////           )]
////        public double oneStep(double currentTime, double stepSize, bool reset)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            if (reset)
////                cvode.reStart(currentTime, model);
////            return cvode.OneStep(currentTime, stepSize);
////        }
////
////
////        // ---------------------------------------------------------------------
////        // Start of Level 3 API Methods
////        // ---------------------------------------------------------------------
////
////        /*[Help("Compute the steady state of the model, returns the sum of squares of the solution")]
////        public double steadyState () {
////            try {
////                if (modelLoaded) {
////                    kinSolver = new kinSolverInterface(model);
////                    return kinSolver.solve(model.y);
////                } else throw new SBWApplicationException (emptyModelStr);
////            } catch (SBWApplicationException) {
////                throw;
////            } catch (Exception e) {
////                throw new SBWApplicationException ("Unexpected error from steadyState()", e.Message);
////            }
////        }*/
////
////
////        //public static void TestSettings()
////        //{
////        //    var rr = new RoadRunner();
////
////        //    Debug.WriteLine(rr.getCapabilities());
////
////        //    rr.UseKinsol = 1;
////        //    Debug.WriteLine(rr.getCapabilities());
////
////        //    rr.setCapabilities(rr.getCapabilities());
////
////        //    rr.UseKinsol = 0;
////        //    Debug.WriteLine(rr.getCapabilities());
////
////        //}
////
////        [Help("Compute the steady state of the model, returns the sum of squares of the solution")]
////        public double steadyState()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            try
////            {
////                if (UseKinsol == 0)
////                    steadyStateSolver = new NLEQInterface(model);
////                else
////                    steadyStateSolver = new KinSolveInterface(model);
////                //oneStep(0.0,0.05);
////                double ss = steadyStateSolver.solve(model.amounts);
////                model.convertToConcentrations();
////                return ss;
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from steadyState solver:", e.Message);
////            }
////        }
////
////
////        // ******************************************************************** }
////        // Multiply matrix 'm1' by 'm2' to give result in Self                  }
////        //                                                                      }
////        // Usage:  A.mult (A1, A2); multiply A1 by A2 giving A                  }
////        //                                                                      }
////        // ******************************************************************** }
////        [Ignore()]
////        public double[][] mult(double[][] m1, double[][] m2)
////        {
////            int m1_nRows = m1.GetLength(0);
////            int m2_nRows = m2.GetLength(0);
////
////            int m1_nColumns = 0;
////            int m2_nColumns = 0;
////
////            if (m1_nRows > 0)
////                m1_nColumns = m1[0].GetLength(0);
////
////            if (m2_nRows > 0)
////                m2_nColumns = m2[0].GetLength(0);
////
////            if (m1.Length == 0)
////                return m1;
////            if (m2.Length == 0)
////                return m2;
////
////            if (m1_nColumns == m2_nRows)
////            {
////                var result = new double[m1_nRows][];
////                for (int i = 0; i < m1_nRows; i++) result[i] = new double[m2_nColumns];
////
////                for (int i = 0; i < result.GetLength(0); i++)
////                    for (int j = 0; j < m2_nColumns; j++)
////                    {
////                        double sum = 0.0;
////                        for (int k = 0; k < m1_nColumns; k++)
////                            sum = sum + (m1[i][k] * m2[k][j]);
////                        result[i][j] = sum;
////                    }
////                return result;
////            }
////
////            if (m1_nRows == m2_nColumns)
////            {
////                return mult(m2, m1);
////            }
////
////            throw new SBWApplicationException("Incompatible matrix operands to multiply");
////        }
////
////
////        [Help("Compute the reduced Jacobian at the current operating point")]
////        public double[][] getReducedJacobian()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] uelast = getUnscaledElasticityMatrix();
////
////                    //                    double[][] Nr = StructAnalysis.getNrMatrix();
////                    //                    double[][] L = StructAnalysis.getLinkMatrix();
////
////                    double[][] I1 = mult(_Nr, uelast);
////                    return mult(I1, _L);
////                }
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from fullJacobian()", e.Message);
////            }
////        }
////
////
////        [Help("Compute the full Jacobian at the current operating point")]
////        public double[][] getFullJacobian()
////        {
////            try
////            {
////                if (modelLoaded)
////                {
////                    double[][] uelast = getUnscaledElasticityMatrix();
////                    //                    double[][] N = StructAnalysis.getReorderedStoichiometryMatrix();
////
////                    return mult(_N, uelast);
////                }
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from fullJacobian()", e.Message);
////            }
////        }
////
////
////        // ---------------------------------------------------------------------
////        // Start of Level 4 API Methods
////        // ---------------------------------------------------------------------
////
////        [Help("Returns the Link Matrix for the currently loaded model")]
////        public double[][] getLinkMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////
////                    return _L; //StructAnalysis.getLinkMatrix();
////
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getLMatrix()", e.Message);
////            }
////        }
////
////        [Help("Returns the reduced stoichiometry matrix (Nr) for the currently loaded model")]
////        public double[][] getNrMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////
////                    return _Nr; //StructAnalysis.getNrMatrix();
////
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getNrMatrix()", e.Message);
////            }
////        }
////
////        [Help("Returns the L0 matrix for the currently loaded model")]
////        public double[][] getL0Matrix()
////        {
////            try
////            {
////                if (modelLoaded)
////
////                    return _L0; // StructAnalysis.getL0Matrix();
////
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getL0Matrix()", e.Message);
////            }
////        }
////
////        [Help("Returns the stoichiometry matrix for the currently loaded model")]
////        public double[][] getStoichiometryMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////
////                    return _N; //StructAnalysis.getReorderedStoichiometryMatrix();
////
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getReorderedStoichiometryMatrix()", e.Message);
////            }
////        }
////
////        [Help("Returns the conservation matrix (gamma) for the currently loaded model")]
////        public double[][] getConservationMatrix()
////        {
////            try
////            {
////                if (modelLoaded)
////                    return StructAnalysis.GetGammaMatrix();
////                //return StructAnalysis.getConservationLawArray();
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getConservationLawArray()", e.Message);
////            }
////        }
////
////        [Help("Returns the number of dependent species in the model")]
////        public int getNumberOfDependentSpecies()
////        {
////            try
////            {
////                if (modelLoaded)
////                    return StructAnalysis.GetNumDependentSpecies();
////                //return StructAnalysis.getNumDepSpecies();
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getNumberOfDependentSpecies()", e.Message);
////            }
////        }
////
////        [Help("Returns the number of independent species in the model")]
////        public int getNumberOfIndependentSpecies()
////        {
////            try
////            {
////                if (modelLoaded)
////                    return StructAnalysis.GetNumIndependentSpecies();
////                //return StructAnalysis.getNumIndSpecies();
////                throw new SBWApplicationException(emptyModelStr);
////            }
////            catch (SBWException)
////            {
////                throw;
////            }
////            catch (Exception e)
////            {
////                throw new SBWApplicationException("Unexpected error from getNumberOfIndependentSpecies()", e.Message);
////            }
////        }
////
////        [Ignore]
////        private double getVariableValue(TVariableType variableType, int variableIndex)
////        {
////            switch (variableType)
////            {
////                case TVariableType.vtFlux:
////                    return model.rates[variableIndex];
////
////                case TVariableType.vtSpecies:
////                    return model.y[variableIndex];
////
////                default:
////                    throw new SBWException("Unrecognised variable in getVariableValue");
////            }
////        }
////
////        [Ignore]
////        private void setParameterValue(TParameterType parameterType, int parameterIndex, double value)
////        {
////            switch (parameterType)
////            {
////                case TParameterType.ptBoundaryParameter:
////                    model.bc[parameterIndex] = value;
////                    break;
////
////                case TParameterType.ptGlobalParameter:
////                    model.gp[parameterIndex] = value;
////                    break;
////
////                case TParameterType.ptFloatingSpecies:
////                    model.y[parameterIndex] = value;
////                    break;
////
////                case TParameterType.ptConservationParameter:
////                    model.ct[parameterIndex] = value;
////                    break;
////
////                case TParameterType.ptLocalParameter:
////                    throw new SBWException("Local parameters not permitted in setParameterValue (getCC, getEE)");
////            }
////        }
////
////        [Ignore]
////        private double getParameterValue(TParameterType parameterType, int parameterIndex)
////        {
////            switch (parameterType)
////            {
////                case TParameterType.ptBoundaryParameter:
////                    return model.bc[parameterIndex];
////
////                case TParameterType.ptGlobalParameter:
////                    return model.gp[parameterIndex];
////
////                // Used when calculating elasticities
////                case TParameterType.ptFloatingSpecies:
////                    return model.y[parameterIndex];
////                case TParameterType.ptConservationParameter:
////                    return model.ct[parameterIndex];
////                case TParameterType.ptLocalParameter:
////                    throw new SBWException("Local parameters not permitted in getParameterValue (getCC?)");
////
////                default:
////                    return 0.0;
////            }
////        }
////
////        /// <summary>
////        /// Fills the second argument with the Inverse of the first argument
////        /// </summary>
////        /// <param name="T2">The Matrix to calculate the Inverse for</param>
////        /// <param name="Inv">will be overriden wiht the inverse of T2 (must already be allocated)</param>
////        private static void GetInverse(Matrix T2, Matrix Inv)
////        {
////            try
////            {
////                Complex[][] T8 = LA.GetInverse(ConvertComplex(T2.data));
////                for (int i1 = 0; i1 < Inv.nRows; i1++)
////                {
////                    for (int j1 = 0; j1 < Inv.nCols; j1++)
////                    {
////                        Inv[i1, j1].Real = T8[i1][j1].Real;
////                        Inv[i1, j1].Imag = T8[i1][j1].Imag;
////                    }
////                }
////            }
////            catch (Exception)
////            {
////                throw new SBWApplicationException("Could not calculate the Inverse");
////            }
////        }
////
////        [Help(
////            "Derpar Continuation, stepSize = stepsize; independentVariable = index to parameter; parameterType = {'globalParameter', 'boundarySpecies'"
////            )]
////        public void computeContinuation(double stepSize, int independentVariable, string parameterTypeStr)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            var derpar = new TDerpar(this, model.getNumTotalVariables, model.getNumIndependentVariables);
////            derpar.setup(model.amounts);
////            switch (parameterTypeStr)
////            {
////                case "globalParameter":
////                    model.amounts =
////                        (double[])
////                        derpar.evalOneStep(model.amounts, stepSize, independentVariable, TDerpar.GLOBAL_PARAMETER_TYPE).
////                            Clone();
////                    break;
////                case "boundarySpecies":
////                    model.amounts =
////                        (double[])
////                        derpar.evalOneStep(model.amounts, stepSize, independentVariable, TDerpar.BOUNDARY_SPECIES_TYPE).
////                            Clone();
////                    break;
////            }
////        }
////
////        [Help("Returns the Symbols of all Flux Control Coefficients.")]
////        public ArrayList getFluxControlCoefficientNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList oReactions = getReactionNames();
////            ArrayList oParameters = ModelGenerator.Instance.getGlobalParameterList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oConservation = ModelGenerator.Instance.getConservationList();
////
////            foreach (string s in oReactions)
////            {
////                var oCCReaction = new ArrayList();
////                var oInner = new ArrayList();
////                oCCReaction.Add(s);
////
////                foreach (string sParameter in oParameters)
////                {
////                    oInner.Add("CC:" + s + "," + sParameter);
////                }
////
////                foreach (string sBoundary in oBoundary)
////                {
////                    oInner.Add("CC:" + s + "," + sBoundary);
////                }
////
////                foreach (string sConservation in oConservation)
////                {
////                    oInner.Add("CC:" + s + "," + sConservation);
////                }
////
////                oCCReaction.Add(oInner);
////                oResult.Add(oCCReaction);
////            }
////
////            return oResult;
////        }
////
////        [Help("Returns the Symbols of all Concentration Control Coefficients.")]
////        public ArrayList getConcentrationControlCoefficientNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList oFloating = getFloatingSpeciesNames();
////            ArrayList oParameters = ModelGenerator.Instance.getGlobalParameterList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oConservation = ModelGenerator.Instance.getConservationList();
////
////            foreach (string s in oFloating)
////            {
////                var oCCFloating = new ArrayList();
////                var oInner = new ArrayList();
////                oCCFloating.Add(s);
////
////                foreach (string sParameter in oParameters)
////                {
////                    oInner.Add("CC:" + s + "," + sParameter);
////                }
////
////                foreach (string sBoundary in oBoundary)
////                {
////                    oInner.Add("CC:" + s + "," + sBoundary);
////                }
////
////                foreach (string sConservation in oConservation)
////                {
////                    oInner.Add("CC:" + s + "," + sConservation);
////                }
////
////                oCCFloating.Add(oInner);
////                oResult.Add(oCCFloating);
////            }
////
////            return oResult;
////        }
////
////        [Help("Returns the Symbols of all Unscaled Concentration Control Coefficients.")]
////        public ArrayList getUnscaledConcentrationControlCoefficientNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList oFloating = getFloatingSpeciesNames();
////            ArrayList oParameters = ModelGenerator.Instance.getGlobalParameterList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oConservation = ModelGenerator.Instance.getConservationList();
////
////            foreach (string s in oFloating)
////            {
////                var oCCFloating = new ArrayList();
////                var oInner = new ArrayList();
////                oCCFloating.Add(s);
////
////                foreach (string sParameter in oParameters)
////                {
////                    oInner.Add("uCC:" + s + "," + sParameter);
////                }
////
////                foreach (string sBoundary in oBoundary)
////                {
////                    oInner.Add("uCC:" + s + "," + sBoundary);
////                }
////
////                foreach (string sConservation in oConservation)
////                {
////                    oInner.Add("uCC:" + s + "," + sConservation);
////                }
////
////                oCCFloating.Add(oInner);
////                oResult.Add(oCCFloating);
////            }
////
////            return oResult;
////        }
////
////        [Help("Returns the Symbols of all Elasticity Coefficients.")]
////        public ArrayList getElasticityCoefficientNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList reactionNames = getReactionNames();
////            ArrayList floatingSpeciesNames = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList boundarySpeciesNames = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList conservationNames = ModelGenerator.Instance.getConservationList();
////            ArrayList globalParameterNames = ModelGenerator.Instance.getGlobalParameterList();
////
////            foreach (string s in reactionNames)
////            {
////                var oCCReaction = new ArrayList();
////                var oInner = new ArrayList();
////                oCCReaction.Add(s);
////
////                foreach (string variable in floatingSpeciesNames)
////                {
////                    oInner.Add(String.Format("EE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in boundarySpeciesNames)
////                {
////                    oInner.Add(String.Format("EE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in globalParameterNames)
////                {
////                    oInner.Add(String.Format("EE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in conservationNames)
////                {
////                    oInner.Add(String.Format("EE:{0},{1}", s, variable));
////                }
////
////                oCCReaction.Add(oInner);
////                oResult.Add(oCCReaction);
////            }
////
////            return oResult;
////        }
////
////        [Help("Returns the Symbols of all Unscaled Elasticity Coefficients.")]
////        public ArrayList getUnscaledElasticityCoefficientNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList oReactions = getReactionNames();
////            ArrayList oFloating = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oGlobalParameters = ModelGenerator.Instance.getGlobalParameterList();
////            ArrayList oConservation = ModelGenerator.Instance.getConservationList();
////
////            foreach (string s in oReactions)
////            {
////                var oCCReaction = new ArrayList();
////                var oInner = new ArrayList();
////                oCCReaction.Add(s);
////
////                foreach (string variable in oFloating)
////                {
////                    oInner.Add(String.Format("uEE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in oBoundary)
////                {
////                    oInner.Add(String.Format("uEE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in oGlobalParameters)
////                {
////                    oInner.Add(String.Format("uEE:{0},{1}", s, variable));
////                }
////
////                foreach (string variable in oConservation)
////                {
////                    oInner.Add(String.Format("uEE:{0},{1}", s, variable));
////                }
////
////
////                oCCReaction.Add(oInner);
////                oResult.Add(oCCReaction);
////            }
////
////            return oResult;
////        }
////
////        [Help("Returns the Symbols of all Floating Species Eigenvalues.")]
////        public ArrayList getEigenValueNames()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            ArrayList oFloating = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////
////            foreach (string s in oFloating)
////            {
////                oResult.Add("eigen_" + s);
////            }
////
////            return oResult;
////        }
////
////        [Help(
////            "Returns symbols of the currently loaded model, that can be used for steady state analysis. Format: array of arrays  { { \"groupname\", { \"item1\", \"item2\" ... } } }  or { { \"groupname\", { \"subgroup\", { \"item1\" ... } } } }."
////            )]
////        public ArrayList getAvailableSteadyStateSymbols()
////        {
////            var oResult = new ArrayList();
////            if (!modelLoaded) return oResult;
////
////            oResult.Add(new ArrayList(new object[] { "Floating Species", getFloatingSpeciesNames() }));
////            oResult.Add(new ArrayList(new object[] { "Boundary Species", getBoundarySpeciesNames() }));
////            oResult.Add(new ArrayList(new object[] { "Floating Species (amount)", getFloatingSpeciesAmountNames() }));
////            oResult.Add(new ArrayList(new object[] { "Boundary Species (amount)", getBoundarySpeciesAmountNames() }));
////            oResult.Add(new ArrayList(new object[] { "Global Parameters", getParameterNames() }));
////            oResult.Add(new ArrayList(new object[] { "Volumes", ModelGenerator.Instance.getCompartmentList() }));
////            oResult.Add(new ArrayList(new object[] { "Fluxes", getReactionNames() }));
////            oResult.Add(new ArrayList(new object[] { "Flux Control Coefficients", getFluxControlCoefficientNames() }));
////            oResult.Add(
////                new ArrayList(new object[] { "Concentration Control Coefficients", getConcentrationControlCoefficientNames() }));
////            oResult.Add(
////                new ArrayList(new object[]
////                                  {
////                                      "Unscaled Concentration Control Coefficients",
////                                      getUnscaledConcentrationControlCoefficientNames()
////                                  }));
////            oResult.Add(new ArrayList(new object[] { "Elasticity Coefficients", getElasticityCoefficientNames() }));
////            oResult.Add(
////                new ArrayList(new object[] { "Unscaled Elasticity Coefficients", getUnscaledElasticityCoefficientNames() }));
////            oResult.Add(new ArrayList(new object[] { "Eigenvalues", getEigenValueNames() }));
////
////            return oResult;
////        }
////
////        [Help("Returns the selection list as returned by computeSteadyStateValues().")]
////        public ArrayList getSteadyStateSelectionList()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if (_oSteadyStateSelection == null)
////            {
////                // default should be species only ...
////                ArrayList floatingSpecies = getFloatingSpeciesNames();
////                _oSteadyStateSelection = new TSelectionRecord[floatingSpecies.Count];
////                for (int i = 0; i < floatingSpecies.Count; i++)
////                {
////                    _oSteadyStateSelection[i] = new TSelectionRecord
////                                                    {
////                                                        selectionType = TSelectionType.clFloatingSpecies,
////                                                        p1 = (string) floatingSpecies[i],
////                                                        index = i
////                                                    };
////                }
////            }
////
////            ArrayList oFloating = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList oBoundary = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList oFluxes = ModelGenerator.Instance.getReactionNames();
////            ArrayList oVolumes = ModelGenerator.Instance.getCompartmentList();
////            ArrayList oRates = getRateOfChangeNames();
////            ArrayList oParameters = getParameterNames();
////
////            var result = new ArrayList();
////            foreach (var record in _oSteadyStateSelection)
////            {
////                switch (record.selectionType)
////                {
////                    case TSelectionType.clTime:
////                        result.Add("time");
////                        break;
////                    case TSelectionType.clBoundaryAmount:
////                        result.Add(string.Format("[{0}]", oBoundary[record.index]));
////                        break;
////                    case TSelectionType.clBoundarySpecies:
////                        result.Add(oBoundary[record.index]);
////                        break;
////                    case TSelectionType.clFloatingAmount:
////                        result.Add("[" + (string)oFloating[record.index] + "]");
////                        break;
////                    case TSelectionType.clFloatingSpecies:
////                        result.Add(oFloating[record.index]);
////                        break;
////                    case TSelectionType.clVolume:
////                        result.Add(oVolumes[record.index]);
////                        break;
////                    case TSelectionType.clFlux:
////                        result.Add(oFluxes[record.index]);
////                        break;
////                    case TSelectionType.clRateOfChange:
////                        result.Add(oRates[record.index]);
////                        break;
////                    case TSelectionType.clParameter:
////                        result.Add(oParameters[record.index]);
////                        break;
////                    case TSelectionType.clEigenValue:
////                        result.Add("eigen_" + record.p1);
////                        break;
////                    case TSelectionType.clElasticity:
////                        result.Add("EE:" + record.p1 + "," + record.p2);
////                        break;
////                    case TSelectionType.clUnscaledElasticity:
////                        result.Add("uEE:" + record.p1 + "," + record.p2);
////                        break;
////                    case TSelectionType.clUnknown:
////                        result.Add(record.p1);
////                        break;
////                }
////
////            }
////
////            return result ;
////        }
////
////        private TSelectionRecord[] GetSteadyStateSelection(ArrayList newSelectionList)
////        {
////            var steadyStateSelection = new TSelectionRecord[newSelectionList.Count];
////            ArrayList fs = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            ArrayList bs = ModelGenerator.Instance.getBoundarySpeciesList();
////            ArrayList rs = ModelGenerator.Instance.getReactionNames();
////            ArrayList vol = ModelGenerator.Instance.getCompartmentList();
////            ArrayList gp = ModelGenerator.Instance.getGlobalParameterList();
////
////            for (int i = 0; i < newSelectionList.Count; i++)
////            {
////                bool set = false;
////                // Check for species
////                for (int j = 0; j < fs.Count; j++)
////                {
////                    if ((string)newSelectionList[i] == (string)fs[j])
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clFloatingSpecies;
////                        set = true;
////                        break;
////                    }
////
////                    if ((string)newSelectionList[i] == "[" + (string)fs[j] + "]")
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clFloatingAmount;
////                        set = true;
////                        break;
////                    }
////
////                    // Check for species rate of change
////                    if ((string)newSelectionList[i] == (string)fs[j] + "'")
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clRateOfChange;
////                        set = true;
////                        break;
////                    }
////                }
////
////                if (set) continue;
////
////                // Check fgr boundary species
////                for (int j = 0; j < bs.Count; j++)
////                {
////                    if ((string)newSelectionList[i] == (string)bs[j])
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clBoundarySpecies;
////                        set = true;
////                        break;
////                    }
////                    if ((string)newSelectionList[i] == "[" + (string)bs[j] + "]")
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clBoundaryAmount;
////                        set = true;
////                        break;
////                    }
////                }
////
////                if (set) continue;
////
////                if ((string)newSelectionList[i] == "time")
////                {
////                    steadyStateSelection[i].selectionType = TSelectionType.clTime;
////                    set = true;
////                }
////
////                for (int j = 0; j < rs.Count; j++)
////                {
////                    // Check for reaction rate
////                    if ((string)newSelectionList[i] == (string)rs[j])
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clFlux;
////                        set = true;
////                        break;
////                    }
////                }
////
////                for (int j = 0; j < vol.Count; j++)
////                {
////                    // Check for volume
////                    if ((string)newSelectionList[i] == (string)vol[j])
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clVolume;
////                        set = true;
////                        break;
////                    }
////                }
////
////                for (int j = 0; j < gp.Count; j++)
////                {
////                    // Check for volume
////                    if ((string)newSelectionList[i] == (string)gp[j])
////                    {
////                        steadyStateSelection[i].index = j;
////                        steadyStateSelection[i].selectionType = TSelectionType.clParameter;
////                        set = true;
////                        break;
////                    }
////                }
////
////                if (set) continue;
////
////                // it is another symbol
////                steadyStateSelection[i].selectionType = TSelectionType.clUnknown;
////                steadyStateSelection[i].p1 = (string)newSelectionList[i];
////            }
////            return steadyStateSelection;
////        }
////
////        [Help("sets the selection list as returned by computeSteadyStateValues().")]
////        public void setSteadyStateSelectionList(ArrayList newSelectionList)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            TSelectionRecord[] steadyStateSelection = GetSteadyStateSelection(newSelectionList);
////
////            _oSteadyStateSelection = steadyStateSelection;
////
////        }
////
////        [Help("performs steady state analysis, returning values as given by setSteadyStateSelectionList().")]
////        public double[] computeSteadyStateValues()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return computeSteadyStateValues(_oSteadyStateSelection, true);
////        }
////
////        private double[] computeSteadyStateValues(TSelectionRecord[] oSelection, bool computeSteadyState)
////        {
////            if (computeSteadyState) steadyState();
////
////            var oResult = new double[oSelection.Length];
////            for (int i = 0; i < oResult.Length; i++)
////            {
////                oResult[i] = computeSteadyStateValue(oSelection[i]);
////            }
////            return oResult;
////
////        }
////
////        [Help("performs steady state analysis, returning values as specified by the given selection list.")]
////        public double[] computeSteadyStateValues(ArrayList oSelection)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            var selection = GetSteadyStateSelection(oSelection);
////            return computeSteadyStateValues(selection, true);
////        }
////
////        private double computeSteadyStateValue(TSelectionRecord record)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            if (record.selectionType == TSelectionType.clUnknown)
////                return computeSteadyStateValue(record.p1);
////            return GetValueForRecord(record);
////        }
////
////        [Help("Returns the value of the given steady state identifier.")]
////        public double computeSteadyStateValue(string sId)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if (sId.StartsWith("CC:"))
////            {
////                string sList = sId.Substring("CC:".Length);
////                string sVariable = sList.Substring(0, sList.IndexOf(","));
////                string sParameter = sList.Substring(sVariable.Length + 1);
////                return getCC(sVariable, sParameter);
////            }
////            if (sId.StartsWith("uCC:"))
////            {
////                string sList = sId.Substring("uCC:".Length);
////                string sVariable = sList.Substring(0, sList.IndexOf(","));
////                string sParameter = sList.Substring(sVariable.Length + 1);
////                return getuCC(sVariable, sParameter);
////            }
////            if (sId.StartsWith("EE:"))
////            {
////                string sList = sId.Substring("EE:".Length);
////                string sReaction = sList.Substring(0, sList.IndexOf(","));
////                string sVariable = sList.Substring(sReaction.Length + 1);
////                return getEE(sReaction, sVariable);
////            }
////            else if (sId.StartsWith("uEE:"))
////            {
////                string sList = sId.Substring("uEE:".Length);
////                string sReaction = sList.Substring(0, sList.IndexOf(","));
////                string sVariable = sList.Substring(sReaction.Length + 1);
////                return getuEE(sReaction, sVariable);
////            }
////            else
////            {
////                if (sId.StartsWith("eigen_"))
////                {
////                    string sSpecies = sId.Substring("eigen_".Length);
////                    int nIndex;
////                    if (ModelGenerator.Instance.floatingSpeciesConcentrationList.find(sSpecies, out nIndex))
////                    {
////                        //SBWComplex[] oComplex = SBW_CLAPACK.getEigenValues(getReducedJacobian());
////                        Complex[] oComplex = LA.GetEigenValues(getReducedJacobian());
////                        if (oComplex.Length > nIndex)
////                        {
////                            return oComplex[nIndex].Real;
////                        }
////                        return Double.NaN;
////                    }
////                    throw new SBWApplicationException(String.Format("Found unknown floating species '{0}' in computeSteadyStateValue()", sSpecies));
////                }
////                try
////                {
////                    return getValue(sId);
////                }
////                catch (Exception )
////                {
////                    throw new SBWApplicationException(String.Format("Found unknown symbol '{0}' in computeSteadyStateValue()", sId));
////                }
////
////            }
////        }
////
////        [Help("Returns the values selected with setSelectionList() for the current model time / timestep")]
////        public double[] getSelectedValues()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            var result = new double[selectionList.Length];
////
////            for (int j = 0; j < selectionList.Length; j++)
////            {
////                result[j] = GetNthSelectedOutput(j, model.time);
////            }
////            return result;
////        }
////
////        [Help("Returns any warnings that occured during the loading of the SBML")]
////        public string[] getWarnings()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return model.Warnings.ToArray();
////        }
////
////        [Help("When turned on, this method will cause rates, event assignments, rules and such to be multiplied " +
////              "with the compartment volume, if species are defined as initialAmounts. By default this behavior is off.")
////        ]
////        public static void ReMultiplyCompartments(bool bValue)
////        {
////            _ReMultiplyCompartments = bValue;
////        }
////
////        [Help("This method turns on / off the computation and adherence to conservation laws."
////              + "By default roadRunner will discover conservation cycles and reduce the model accordingly.")]
////        public static void ComputeAndAssignConservationLaws(bool bValue)
////        {
////            _bComputeAndAssignConservationLaws = bValue;
////        }
////
////        [Help("Returns the current generated source code")]
////        public string getCSharpCode()
////        {
////            if (modelLoaded)
////            {
////                return _sModelCode;
////            }
////
////            throw new SBWApplicationException("Model has to be loaded first");
////        }
////
////        [Help(
////            "Performs a steady state parameter scan with the given parameters returning all elments from the selectionList: (Format: symnbol, startValue, endValue, stepSize)"
////            )]
////        public double[][] steadyStateParameterScan(string symbol, double startValue, double endValue, double stepSize)
////        {
////            var results = new List<double[]>();
////
////            double initialValue = getValue(symbol);
////            double current = startValue;
////
////            while (current < endValue)
////            {
////                setValue(symbol, current);
////                try
////                {
////                    steadyState();
////                }
////                catch (Exception)
////                {
////                    //
////                }
////
////                var currentRow = new List<double> {current};
////                currentRow.AddRange(getSelectedValues());
////
////                results.Add(currentRow.ToArray());
////                current += stepSize;
////            }
////            setValue(symbol, initialValue);
////
////            return results.ToArray();
////        }
////
////
////        [Help("Returns the SBML with the current parameterset")]
////        public string writeSBML()
////        {
////            NOM.loadSBML(NOM.getParamPromotedSBML(sbmlStr));
////            var state = new ModelState(model);
////
////            ArrayList array = getFloatingSpeciesNames();
////            for (int i = 0; i < array.Count; i++)
////                NOM.setValue((string)array[i], state.FloatingSpeciesConcentrations[i]);
////
////            array = getBoundarySpeciesNames();
////            for (int i = 0; i < array.Count; i++)
////                NOM.setValue((string)array[i], state.BoundarySpeciesConcentrations[i]);
////
////            array = getCompartmentNames();
////            for (int i = 0; i < array.Count; i++)
////                NOM.setValue((string)array[i], state.CompartmentVolumes[i]);
////
////            array = getGlobalParameterNames();
////            for (int i = 0; i < Math.Min(array.Count, state.GlobalParameters.Length); i++)
////                NOM.setValue((string)array[i], state.GlobalParameters[i]);
////
////            return NOM.getSBML();
////        }
////
////        #region Get Local Parameter Names / Values
////
////        // -----------------------------------------------------------------
////
////        [Help("Get the number of local parameters for a given reaction")]
////        public int getNumberOfLocalParameters(int reactionId)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return getNumberOfLocalParameters(reactionId);
////        }
////
////        [Help("Sets the value of a global parameter by its index")]
////        public void setLocalParameterByIndex(int reactionId, int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if ((reactionId >= 0) && (reactionId < model.getNumReactions) &&
////                (index >= 0) && (index < model.getNumLocalParameters(reactionId)))
////                model.lp[reactionId][index] = value;
////            else
////                throw new SBWApplicationException(string.Format("Index in setLocalParameterByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns the value of a global parameter by its index")]
////        public double getLocalParameterByIndex(int reactionId, int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((reactionId >= 0) && (reactionId < model.getNumReactions) &&
////                (index >= 0) && (index < model.getNumLocalParameters(reactionId)))
////                return model.lp[reactionId][index];
////
////            throw new SBWApplicationException(String.Format("Index in getLocalParameterByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Set the values for all global parameters in the model")]
////        public void setLocalParameterValues(int reactionId, double[] values)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////
////            if ((reactionId >= 0) && (reactionId < model.getNumReactions))
////                model.lp[reactionId] = values;
////            else
////                throw new SBWApplicationException(String.Format("Index in setLocalParameterValues out of range: [{0}]", reactionId));
////        }
////
////        [Help("Get the values for all global parameters in the model")]
////        public double[] getLocalParameterValues(int reactionId)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((reactionId >= 0) && (reactionId < model.getNumReactions))
////                return model.lp[reactionId];
////            throw new SBWApplicationException(String.Format("Index in getLocalParameterValues out of range: [{0}]", reactionId));
////        }
////
////        [Help("Gets the list of parameter names")]
////        public ArrayList getLocalParameterNames(int reactionId)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((reactionId >= 0) && (reactionId < model.getNumReactions))
////                return ModelGenerator.Instance.getLocalParameterList(reactionId);
////            throw (new SBWApplicationException("reaction Id out of range in call to getLocalParameterNames"));
////        }
////
////        [Help("Returns a list of global parameter tuples: { {parameter Name, value},...")]
////        public ArrayList getAllLocalParameterTupleList()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            var tupleList = new ArrayList();
////            for (int i = 0; i < ModelGenerator.Instance.getNumberOfReactions(); i++)
////            {
////                var tuple = new ArrayList();
////                ArrayList lpList = ModelGenerator.Instance.getLocalParameterList(i);
////                tuple.Add(i);
////                for (int j = 0; j < lpList.Count; j++)
////                {
////                    tuple.Add(lpList[j]);
////                    tuple.Add(model.lp[i][j]);
////                }
////                tupleList.Add(tuple);
////            }
////            return tupleList;
////        }
////
////        #endregion
////
////        #region Get Reaction Rate / Names ...
////
////        // -----------------------------------------------------------------
////
////        [Help("Get the number of reactions")]
////        public int getNumberOfReactions()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return model.getNumReactions;
////        }
////
////        [Help("Returns the rate of a reaction by its index")]
////        public double getReactionRate(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < model.getNumReactions))
////            {
////                model.convertToConcentrations();
////                model.computeReactionRates(0.0, model.y);
////                return model.rates[index];
////            }
////            throw new SBWApplicationException(String.Format("Index in getReactionRate out of range: [{0}]", index));
////        }
////
////        [Help("Returns the rate of changes of a species by its index")]
////        public double getRateOfChange(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < model.getNumTotalVariables))
////            {
////                model.computeAllRatesOfChange();
////                return model.dydt[index];
////            }
////            throw new SBWApplicationException(String.Format("Index in getRateOfChange out of range: [{0}]", index));
////        }
////
////        [Help("Returns the names given to the rate of change of the floating species")]
////        public ArrayList getRateOfChangeNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            ArrayList sp = ModelGenerator.Instance.getFloatingSpeciesConcentrationList(); // Reordered list
////            for (int i = 0; i < sp.Count; i++)
////                sp[i] = sp[i] + "'";
////            return sp;
////        }
////
////        [Help("Returns the rates of changes given an array of new floating species concentrations")]
////        public double[] getRatesOfChangeEx(double[] values)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            model.y = values;
////            model.evalModel(0.0, BuildModelEvalArgument());
////            return model.dydt;
////        }
////
////        [Help("Returns the rates of changes given an array of new floating species concentrations")]
////        public double[] getReactionRatesEx(double[] values)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            model.computeReactionRates(0.0, values);
////            return model.rates;
////        }
////
////
////        public string[] GetFloatingSpeciesNamesArray()
////        {
////            return (string[])getFloatingSpeciesNames().ToArray(typeof(string));
////        }
////
////        public string[] GetGlobalParameterNamesArray()
////        {
////            return (string[])getGlobalParameterNames().ToArray(typeof(string));
////        }
////
////        #endregion
////
////        #region Get Compartment Names / Values
////
////        [Help("Get the number of compartments")]
////        public int getNumberOfCompartments()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return model.getNumCompartments;
////        }
////
////        [Help("Sets the value of a compartment by its index")]
////        public void setCompartmentByIndex(int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < model.getNumCompartments))
////                model.c[index] = value;
////            else
////                throw new SBWApplicationException(String.Format("Index in getCompartmentByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns the value of a compartment by its index")]
////        public double getCompartmentByIndex(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            if ((index >= 0) && (index < model.getNumCompartments))
////                return model.c[index];
////            throw (new SBWApplicationException(String.Format("Index in getCompartmentByIndex out of range: [{0}]", index)));
////        }
////
////        [Help("Returns the value of a compartment by its index")]
////        public void setCompartmentVolumes(double[] values)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            if (values.Length < model.getNumCompartments)
////                model.c = values;
////            else
////                throw (new SBWApplicationException(String.Format("Size of vector out not in range in setCompartmentValues: [{0}]", values.Length)));
////        }
////
////        [Help("Gets the list of compartment names")]
////        public ArrayList getCompartmentNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return ModelGenerator.Instance.getCompartmentList();
////        }
////
////        #endregion
////
////        #region Get Boundary Species Names / Values
////
////        // -----------------------------------------------------------------
////
////        [Help("Get the number of boundary species")]
////        public int getNumberOfBoundarySpecies()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return model.getNumBoundarySpecies;
////        }
////
////        [Help("Sets the value of a boundary species by its index")]
////        public void setBoundarySpeciesByIndex(int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            if ((index >= 0) && (index < model.getNumBoundarySpecies))
////                model.bc[index] = value;
////            else
////                throw new SBWApplicationException(String.Format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns the value of a boundary species by its index")]
////        public double getBoundarySpeciesByIndex(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            if ((index >= 0) && (index < model.getNumBoundarySpecies))
////                return model.bc[index];
////            throw new SBWApplicationException(String.Format("Index in getBoundarySpeciesByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns an array of boundary species concentrations")]
////        public double[] getBoundarySpeciesConcentrations()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return model.bc;
////        }
////
////        [Help("Set the concentrations for all boundary species in the model")]
////        public void setBoundarySpeciesConcentrations(double[] values)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            model.bc = values;
////        }
////
////        [Help("Gets the list of boundary species names")]
////        public ArrayList getBoundarySpeciesNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return ModelGenerator.Instance.getBoundarySpeciesList();
////        }
////
////        [Help("Gets the list of boundary species amount names")]
////        public ArrayList getBoundarySpeciesAmountNames()
////        {
////            var oResult = new ArrayList();
////            foreach (string s in getBoundarySpeciesNames()) oResult.Add("[" + s + "]");
////            return oResult;
////        }
////
////        #endregion
////
////        #region Get Floating Species Names / Values
////
////        // -----------------------------------------------------------------
////
////        [Help("Get the number of floating species")]
////        public int getNumberOfFloatingSpecies()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return model.getNumTotalVariables;
////        }
////
////        [Help("Sets the value of a floating species by its index")]
////        public void setFloatingSpeciesByIndex(int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < model.getNumTotalVariables))
////            {
////                model.setConcentration(index, value); // This updates the amount vector aswell
////                if (!_bConservedTotalChanged) model.computeConservedTotals();
////            }
////            else
////                throw new SBWApplicationException(String.Format("Index in setFloatingSpeciesByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns the value of a floating species by its index")]
////        public double getFloatingSpeciesByIndex(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            if ((index >= 0) && (index < model.getNumTotalVariables))
////                return model.getConcentration(index);
////            throw new SBWApplicationException(String.Format("Index in getFloatingSpeciesByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Returns an array of floating species concentrations")]
////        public double[] getFloatingSpeciesConcentrations()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            model.convertToConcentrations();
////            return model.y;
////        }
////
////        [Help("returns an array of floating species initial conditions")]
////        public double[] getFloatingSpeciesInitialConcentrations()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return model.init_y;
////        }
////
////
////        // This is a level 1 Method 1
////        [Help("Set the concentrations for all floating species in the model")]
////        public void setFloatingSpeciesConcentrations(double[] values)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            model.y = values;
////            // Update the amounts vector at the same time
////            model.convertToAmounts();
////            if (!_bConservedTotalChanged) model.computeConservedTotals();
////        }
////
////        [Help("Sets the value of a floating species by its index")]
////        public void setFloatingSpeciesInitialConcentrationByIndex(int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < model.init_y.Length))
////            {
////                model.init_y[index] = value;
////                reset();
////            }
////            else
////                throw new SBWApplicationException(String.Format("Index in setFloatingSpeciesInitialConcentrationByIndex out of range: [{0}]", index));
////        }
////
////        [Help("Sets the initial conditions for all floating species in the model")]
////        public void setFloatingSpeciesInitialConcentrations(double[] values)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////
////            model.init_y = values;
////            reset();
////        }
////
////
////        // This is a Level 1 method !
////        [Help("Returns a list of floating species names")]
////        public ArrayList getFloatingSpeciesNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            return ModelGenerator.Instance.getFloatingSpeciesConcentrationList(); // Reordered list
////        }
////
////        [Help("Returns a list of floating species initial condition names")]
////        public ArrayList getFloatingSpeciesInitialConditionNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            ArrayList floatingSpeciesNames = ModelGenerator.Instance.getFloatingSpeciesConcentrationList();
////            var result = new ArrayList();
////            foreach (object item in floatingSpeciesNames)
////            {
////                result.Add(String.Format("init({0})", item));
////            }
////            return result;
////        }
////
////
////        [Help("Returns the list of floating species amount names")]
////        public ArrayList getFloatingSpeciesAmountNames()
////        {
////            var oResult = new ArrayList();
////            foreach (string s in getFloatingSpeciesNames()) oResult.Add(String.Format("[{0}]", s));
////            return oResult;
////        }
////
////        #endregion
////
////        #region Get Global Parameter  Names / Values
////
////        // -----------------------------------------------------------------
////
////        [Help("Get the number of global parameters")]
////        public int getNumberOfGlobalParameters()
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            return ModelGenerator.Instance.getGlobalParameterList().Count;
////        }
////
////        [Help("Sets the value of a global parameter by its index")]
////        public void setGlobalParameterByIndex(int index, double value)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            if ((index >= 0) && (index < model.getNumGlobalParameters + model.ct.Length))
////            {
////                if (index >= model.getNumGlobalParameters)
////                {
////                    model.ct[index - model.getNumGlobalParameters] = value;
////                    model.updateDependentSpeciesValues(model.y);
////                    _bConservedTotalChanged = true;
////                }
////                else
////                    model.gp[index] = value;
////            }
////            else
////                throw new SBWApplicationException(String.Format("Index in getNumGlobalParameters out of range: [{0}]", index));
////        }
////
////        [Help("Returns the value of a global parameter by its index")]
////        public double getGlobalParameterByIndex(int index)
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            if ((index >= 0) && (index < (model.getNumGlobalParameters + model.ct.Length)))
////            {
////                var result = new double[model.gp.Length + model.ct.Length];
////                model.gp.CopyTo(result, 0);
////                model.ct.CopyTo(result, model.gp.Length);
////                return result[index];
////                //return model.gp[index];
////            }
////            throw new SBWApplicationException(String.Format("Index in getNumGlobalParameters out of range: [{0}]", index));
////        }
////
////        [Help("Set the values for all global parameters in the model")]
////        public void setGlobalParameterValues(double[] values)
////        {
////            if (!modelLoaded) throw new SBWApplicationException(emptyModelStr);
////            if (values.Length == model.gp.Length)
////                model.gp = values;
////            else
////            {
////                for (int i = 0; i < model.gp.Length; i++)
////                {
////                    model.gp[i] = values[i];
////                }
////                for (int i = 0; i < model.ct.Length; i++)
////                {
////                    model.gp[i] = values[i + model.gp.Length];
////                    _bConservedTotalChanged = true;
////                }
////                model.updateDependentSpeciesValues(model.y);
////            }
////        }
////
////        [Help("Get the values for all global parameters in the model")]
////        public double[] getGlobalParameterValues()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            if (model.ct.Length > 0)
////            {
////                var result = new double[model.gp.Length + model.ct.Length];
////                model.gp.CopyTo(result, 0);
////                model.ct.CopyTo(result, model.gp.Length);
////                return result;
////            }
////            return model.gp;
////        }
////
////        [Help("Gets the list of parameter names")]
////        public ArrayList getGlobalParameterNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            return ModelGenerator.Instance.getGlobalParameterList();
////        }
////
////        [Help("Returns a list of global parameter tuples: { {parameter Name, value},...")]
////        public ArrayList getAllGlobalParameterTupleList()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////
////            var tupleList = new ArrayList();
////            ArrayList gp = ModelGenerator.Instance.getGlobalParameterList();
////            for (int i = 0; i < gp.Count; i++)
////            {
////                var tuple = new ArrayList {gp[i], model.gp[i]};
////                tupleList.Add(tuple);
////            }
////            return tupleList;
////        }
////
////        private ArrayList getParameterNames()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            ArrayList sp = ModelGenerator.Instance.getGlobalParameterList(); // Reordered list
////            return sp;
////        }
////
////        [Help("Updates the model based on all recent changes")]
////        public void EvalModel()
////        {
////            if (!modelLoaded)
////                throw new SBWApplicationException(emptyModelStr);
////            model.convertToAmounts();
////            model.evalModel(model.time, cvode.BuildEvalArgument());
////        }
////
////        #endregion
////
////        #region Information about Roadrunner: getName
////
////        [Help("Returns the name of module")]
////        public string getName()
////        {
////            return "roadRunner";
////        }
////
////        [Help("Returns the version number of the module")]
////        public static string getVersion()
////        {
////            return "2.0.1";
////        }
////
////        [Help("Returns the name of the module author")]
////        public static string getAuthor()
////        {
////            return "H. M. Sauro and F. T. Bergmann";
////        }
////
////        [Help("Returns a description of the module")]
////        public static string getDescription()
////        {
////            return "Simulator API based on CVODE/NLEQ/CSharp implementation";
////        }
////
////        [Help("Returns the display name of the module")]
////        public static string getDisplayName()
////        {
////            return "RoadRunner";
////        }
////
////        [Help("Returns the copyright string for the module")]
////        public static string getCopyright()
////        {
////            return "(c) 2009 H. M. Sauro and F. T. Bergmann, BSD Licence";
////        }
////
////        [Help("Returns the URL string associated with the module (if any)")]
////        public static string getURL()
////        {
////            return "http://sys-bio.org";
////        }
////
////        #endregion
////
////        #region Nested type: TSelectionRecord
////
////        public struct TSelectionRecord
////        {
////            public int index;
////            public string p1;
////            public string p2;
////
////            public TSelectionType selectionType;
////        }
////
////        #endregion
////
////
//// #if DEBUG
////       public static void TestChange()
////        {
////            var sbml = File.ReadAllText(@"C:\Users\fbergmann\Desktop\testModel.xml");
////            var sim = new RoadRunner();
////            sim.loadSBML(sbml);
////            sim.setTimeStart(0);
////            sim.setTimeEnd(10);
////            sim.setNumPoints(10);
////            var data = sim.simulate();
////            var writer = new StringWriter();
////            DumpResults(writer, data, sim.getSelectionList());
////            sim.changeInitialConditions(new double[] { 20, 0 });
////            sim.reset();
////            data = sim.simulate();
////            writer = new StringWriter();
////            DumpResults(writer, data, sim.getSelectionList());
////        }
////#endif
////    }
////
////
////
////
////}
