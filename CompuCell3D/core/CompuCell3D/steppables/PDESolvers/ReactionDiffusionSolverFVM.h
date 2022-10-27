#ifndef COMPUCELL3DREACTIONDIFFUSIONSOLVERFVM_H
#define COMPUCELL3DREACTIONDIFFUSIONSOLVERFVM_H

#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableVector.h"
#include "FluctuationCompensator.h"

#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

#include "DiffSecrData.h"
#include "BoundaryConditionSpecifier.h"

#include <CompuCell3D/CC3DEvents.h>

#include <string>

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <muParser/muParser.h>

#include "PDESolversDLLSpecifier.h"
/**
 * @brief Template class to perform element wise vector addition
 * 
 *
 */
template<class T>
struct vecPlus {
	typedef std::vector<T> first_argument_type;
	typedef std::vector<T> second_argument_type;
	typedef std::vector<T> result_type;

	/**
	 * @brief Performs element wise vector addition and returns resulting vector
	 * 
	 * @param _lhs First vector of type T
	 * @param _rhs Second vector of type T
	 * @return std::vector<T> contains sum of elements in _lhs and _rhs
	 */
	std::vector<T> operator()(const std::vector<T>& _lhs, const std::vector<T>& _rhs) const {	
		
		std::vector<T> _res = std::vector<T>(_lhs.size(), 0.0);
		for (unsigned int _vecIdx = 0; _vecIdx < _lhs.size(); ++_vecIdx) { _res[_vecIdx] = _lhs[_vecIdx] + _rhs[_vecIdx]; }
		return (_res);
	}
	
};

/**
 * @brief Template class to perform element wise vector multiplication
 * 
 *
 */
template<class T>
struct vecMult {
	typedef std::vector<T> first_argument_type;
	typedef std::vector<T> second_argument_type;
	typedef std::vector<T> result_type;

	/**
	 * @brief Performs element wise vector product and returns resulting vector
	 * 
	 * @param _lhs First vector of type T
	 * @param _rhs Second vector of type T
	 * @return std::vector<T> contains product of elements in _lhs and _rhs
	 */
	std::vector<T> operator()(const std::vector<T>& _lhs, const std::vector<T>& _rhs) const {	
		std::vector<T> _res = std::vector<T>(_lhs.size(), 0.0);
		for (unsigned int _vecIdx = 0; _vecIdx < _lhs.size(); ++_vecIdx) { _res[_vecIdx] = _lhs[_vecIdx] * _rhs[_vecIdx]; }
		return (_res);
	}
};

/**
 * @brief Template class to convert vector to string
 * 
 *
 */
template<class T>
struct vecToString {
	typedef std::vector<T> first_argument_type;
	typedef std::string result_type;
	
	/**
	 * @brief Converts vector to string
	 * 
	 * @param _vec input vector
	 * @return std::string string containing elements of _vec separated by space
	 */
	std::string operator()(const std::vector<T>& _vec) const {
		std::string _res = "";
		for (unsigned int _vecIdx = 0; _vecIdx < _vec.size(); ++_vecIdx) { _res += to_string(_vec[_vecIdx]) + " "; }
		return (_res);
	}
};

/**
 * @brief mu parser class
 * 
 */
namespace mu {

	class Parser; //mu parser class
};

/**
 * @brief assign elements to namespace Compucell3D
 * 
 */
namespace CompuCell3D {

	/**
	@author T.J. Sego, Ph.D.
	*/
	
	class Potts3D;
	class Simulator;
	class Cell;
	class CellInventory;
	class Automaton;

	class PixelTrackerData;
	class PixelTrackerPlugin;

	class DiffusionData;
	class SecretionDataFlex;

    class FluctuationCompensator;

	template <typename Y> class Field3D;
	template <typename Y> class Field3DImpl;
	template <typename Y> class WatchableField3D;

	template <typename Y> class RDFVMField3DWrap;
	class ReactionDiffusionSolverFVM;
	class ReactionDiffusionSolverFV;
	class ReactionDiffusionSolverFVMCellData;

	///////////////////////////////////////////////////////////////// Solver /////////////////////////////////////////////////////////////////
	// Surface indices are ordered +x, -x, +y, -y, +z, -z
	/**
	 * @brief Reaction diffusion solver classs
	 * 
	 */
	class PDESOLVERS_EXPORT ReactionDiffusionSolverFVM :public DiffusableVector<float>
	{

	public:
		typedef void(ReactionDiffusionSolverFVM::*DiffusivityModeInitializer)(unsigned int);
		typedef void(ReactionDiffusionSolverFVM::*FluxConditionInitializer)(unsigned int);

	private:
		ExtraMembersGroupAccessor<ReactionDiffusionSolverFVMCellData> ReactionDiffusionSolverFVMCellDataAccessor;

		PixelTrackerPlugin *pixelTrackerPlugin;
		NeighborTrackerPlugin *neighborTrackerPlugin;

		FluctuationCompensator *fluctuationCompensator;

		std::vector<ReactionDiffusionSolverFV *>  fieldFVs;

		// Need to set these according to lattice type
		std::vector<unsigned int> indexMapSurfToCoord;
		std::vector<int> surfaceNormSign;
		std::map<std::string, unsigned int> surfaceMapNameToIndex;
		std::vector<float> signedDistanceBySurfaceIndex;
		std::vector<float> surfaceAreas;

		std::vector<std::string> availableUnitsTime = { "s", "min", "hr", "day" };
		std::vector<float> availableUnitTimeConv = std::vector<float>{ (float)1.0, (float)(60.0), (float)(3600.0), (float)(86400.0) };

		unsigned int numFields;
		std::map<std::string, unsigned int> fieldNameToIndexMap;
		std::vector<RDFVMField3DWrap<float> *> concentrationFieldVector;
		std::vector<std::string> fieldSymbolsVec;
		std::vector<std::vector<std::string> > fieldExpressionStringsDiag;
		std::vector<std::string> fieldExpressionStringsMergedDiag;
		std::vector<std::vector<std::string> > fieldExpressionStringsOffDiag;
		std::vector<std::string> fieldExpressionStringsMergedOffDiag;
		std::vector<double> constantDiffusionCoefficientsVec;
		std::vector<Field3D<float> *> diffusivityFieldIndexToFieldMap;
		std::vector<bool> diffusivityFieldInitialized;
		std::vector<SecretionData> secrFieldVec;

		std::vector<FluxConditionInitializer> fluxConditionInitializerPtrs;

		std::vector<std::vector<tuple<std::string, float> > > basicBCData;
		std::vector<bool> periodicBoundaryCheckVector;

		unsigned int numCellTypes;
		std::map<std::string, unsigned int> cellTypeNameToIndexMap;
		std::vector<std::vector<double> > constantDiffusionCoefficientsVecCellType;
		std::vector<DiffusivityModeInitializer> diffusivityModeInitializerPtrs;

		double integrationTimeStep;

		std::vector<std::vector<std::vector<double> > > constantPermeationCoefficientsVecCellType;
		std::vector<std::vector<std::vector<double> > > constPermBiasCoeffsVecCellType;
		std::vector<bool> usingSimplePermeableInterfaces;

		bool cellDataLoaded;

		std::string diffusivityFieldSuffixStd = "Diff";
		std::string expressionSuffixStd = "ExpSym";

	protected:

		Simulator *sim;
		Potts3D *potts;
		Automaton *automaton;
		ParallelUtilsOpenMP *pUtils;
		ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

		CC3DXMLElement *xmlData;

		double physTime;
		double incTime;
		string unitsTime = "s";
		float unitTimeConv = 1.0;
		float lengthX;
		float lengthY;
		float lengthZ;
		unsigned int currentStep;

		bool autoTimeSubStep;

		std::vector<double> *fvMaxStableTimeSteps;

		Dim3D fieldDim;

	public:

		ReactionDiffusionSolverFVM();

		virtual ~ReactionDiffusionSolverFVM();

		ExtraMembersGroupAccessor<ReactionDiffusionSolverFVMCellData> * getReactionDiffusionSolverFVMCellDataAccessorPtr() { return & ReactionDiffusionSolverFVMCellDataAccessor; }

		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);
		virtual void extraInit(Simulator *simulator);
		virtual void handleEvent(CC3DEvent & _event);
		// Steppable interface
		virtual void start() {}
		virtual void step(const unsigned int _currentStep);
		virtual void finish() {}

		// Steerable Object interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);
		virtual std::string steerableName();
		virtual std::string toString();

		// Diffusable Vector interface
		Array3DBordersField3DAdapter<float> * getConcentrationField(const std::string & name) { return (Array3DBordersField3DAdapter<float> *) concentrationFieldVector[getFieldIndexByName(name)]; }
		void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) { 
			boundaryStrategy = BoundaryStrategy::getInstance();
			maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
			concentrationFieldNameVector.assign(numberOfFields, std::string());
		}
		std::vector<std::string> getConcentrationFieldNameVector() { return concentrationFieldNameVector; }

		// Interface between Python and FluctuationCompensator

        // Call to update compensator for this solver before next compensation
        // Call this after modifying field values outside of core routine
        virtual void updateFluctuationCompensator() { if (fluctuationCompensator) fluctuationCompensator->updateTotalConcentrations(); }

		// Solver routines
		/**
		 * @brief Initialises cell data and sets cell diffusivity and permeability coefficients
		 * 
		 */
		void loadCellData();
		
		/**
		 * @brief Loads field expression multiplier and field expression independent for each field index
		 * 
		 */
		void loadFieldExpressions();

		/**
		 * @brief Sets field expressions for diagonal functions based on _fieldIndex
		 * 
		 * @param _fieldIndex 
		 */
		void loadFieldExpressionMultiplier(unsigned int _fieldIndex);

		/**
		 * @brief Sets field expressions for off-diagonal functions based on _fieldIndex
		 * 
		 * @param _fieldIndex 
		 */
		void loadFieldExpressionIndependent(unsigned int _fieldIndex);

		/**
		 * @brief Gets _fieldIndex from _fieldName and then sets 
		 *  field expressions for diagonal functions based on _fieldIndex
		 * 
		 * @param _fieldName 
		 */
		virtual void loadFieldExpressionMultiplier(std::string _fieldName) { loadFieldExpressionMultiplier(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 */
		virtual void loadFieldExpressionIndependent(std::string _fieldName) { loadFieldExpressionIndependent(getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief Sets field expression multiplier using _fieldName and _expr
		 *  and then uses _fieldName to fetch _fieldIndex which is then used
		 * to set field expression for diagonal functions
		 * @param _fieldName 
		 * @param _expr 
		 */
		virtual void loadFieldExpressionMultiplier(std::string _fieldName, std::string _expr);
		
		/**
		 * @brief Sets field expression multiplier using _fieldName and _expr
		 *  and then uses _fieldName to fetch _fieldIndex which is then used
		 * to set field expression for off-diagonal functions
		 * 
		 * @param _fieldName 
		 * @param _expr 
		 */
		virtual void loadFieldExpressionIndependent(std::string _fieldName, std::string _expr);

		/**
		 * @brief calls _fv's setDiagonalFunctionExpression method and sets field expression
		 * for diagonal functions based on _fieldIndex and expression string at that field index
		 * 
		 * @param _fieldIndex 
		 * @param _fv 
		 */
		void loadFieldExpressionMultiplier(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv);

		/**
		 * @brief calls _fv's setOffDiagonalFunctionExpression method and sets field expression
		 * for off-diagonal functions based on _fieldIndex and expression string at that field index
		 * 
		 * @param _fieldIndex 
		 * @param _fv 
		 */
		void loadFieldExpressionIndependent(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv);
		
		/**
		 * @brief Creates a lattice with x,y,z dimensions from _fieldDim,
		 * Initialises Finite Volumes, registers field symbols for each 
		 * finite volume, loads field expressions and applies basic boundary conditions 
		 * for z,y and x dimensions 
		 * 
		 * @param _fieldDim 
		 */
		void initializeFVs(Dim3D _fieldDim);

		/**
		 * @brief uses a mu::Parser instance to parse _expr
		 * based on x,y,z values from field dimension and 
		 * sets the concentration at _fieldIndex based on the parsed value
		 * 
		 * @param _fieldIndex 
		 * @param _expr 
		 */
		void initializeFieldUsingEquation(unsigned int _fieldIndex, std::string _expr);

		/**
		 * @brief gets _field Index from _field Name and then
		 * uses a mu::Parser instance to parse _expr
		 * based on x,y,z values from field dimension and 
		 * sets the concentration at _fieldIndex based on the parsed value
		 * 
		 * @param _fieldName 
		 * @param _expr 
		 */
		virtual void initializeFieldUsingEquation(std::string _fieldName, std::string _expr) { initializeFieldUsingEquation(getFieldIndexByName(_fieldName), _expr); }
		
		// FV interface

		/**
		 * @brief Get the Coords Of FV object
		 * 
		 * @param _fv 
		 * @return Point3D 
		 */
		Point3D getCoordsOfFV(ReactionDiffusionSolverFV *_fv);
		
		/**
		 * @brief returns CellG cooresponding to 3D point coodinates from _fv
		 * 
		 * @param _fv 
		 * @return CellG* 
		 */
		CellG * FVtoCellMap(ReactionDiffusionSolverFV * _fv);
		
		/**
		 * @brief Get the Constant Field Diffusivity object at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getConstantFieldDiffusivity(unsigned int _fieldIndex) { return constantDiffusionCoefficientsVec[_fieldIndex]; }

		/**
		 * @brief Get the Diffusivity Field Pt Val object
		 * 
		 * @param _fieldIndex 
		 * @param pt 
		 * @return float 
		 */
		float getDiffusivityFieldPtVal(unsigned int _fieldIndex, const Point3D &pt) { return diffusivityFieldIndexToFieldMap[_fieldIndex]->get(pt); }

		/**
		 * @brief Sets pointers to getConstantDiffusivity function
		 * for fieldDiffusivityFunctionPtrs vector and sets constant diffusivity
		 * derived from constantDiffusionCoefficientsVec for auxVars at _fieldIndex 
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex) { useConstantDiffusivity(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }

		/**
		 * @brief Sets pointers to getConstantDiffusivity function
		 * for fieldDiffusivityFunctionPtrs vector and sets constant diffusivity
		 * using _diffusivityCoefficient for auxVars at _fieldIndex 
		 * 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex, double _diffusivityCoefficient);

		/**
		 @brief Sets pointers to getConstantDiffusivityById function at 
		 * _fieldIndex for fieldDiffusivityFunctionPtrs vector
		 *
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void useConstantDiffusivity(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivity(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }

		/**
		 * @brief Gets _fieldIndex from _fieldName and sets fieldDiffusivityFunctionPtrs at _fieldIndex
		 * for each element of fieldFVS by calling useConstantDiffusivityById using fieldIndex and diffusivity constant
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivityByType(unsigned int _fieldIndex) { useConstantDiffusivityByType(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }

		/**
		 * @brief sets fieldDiffusivityFunctionPtrs at _fieldIndex
		 * for each element of fieldFVS by calling useConstantDiffusivityById using fieldIndex and diffusivity constant
		 * 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void useConstantDiffusivityByType(unsigned int _fieldIndex, double _diffusivityCoefficient);

		/**
		 * @brief sets fieldDiffusivityFunctionPtrs at _fieldIndex
		 * for each element of fieldFVS by calling useConstantDiffusivityById using fieldIndex and diffusivity constant
		 * Function is overridden by derived class
		 * 
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void useConstantDiffusivityByType(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivityByType(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }

		/**
		 * @brief Initialises diffusivity at _fieldIndex and then calls 
		 * useFieldDiffusivityInMedium function for all element in fieldFVS
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityInMedium(unsigned int _fieldIndex);

		/**
		 * @brief gets _fieldIndex from _fieldName and then 
		 * initialises diffusivity at _fieldIndex and then calls 
		 * useFieldDiffusivityInMedium function for all element in fieldFVS
		 * The function is overridden by the derived class
		 * 
		 * @param _fieldName 
		 */
		virtual void useFieldDiffusivityInMedium(std::string _fieldName) { useFieldDiffusivityInMedium(getFieldIndexByName(_fieldName)); }

		/**
		 * @brief Initialises diffusivity at _fieldIndex and then calls 
		 * useFieldDiffusivityEverywhere function for all element in fieldFVS
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityEverywhere(unsigned int _fieldIndex);

		/**
		 * @brief gets _fieldIndex from _fieldName and then 
		 * initialises diffusivity at _fieldIndex and then calls 
		 * useFieldDiffusivityEverywhere function for all element in fieldFVS
		 * The function is overridden by the derived class
		 * 
		 * @param _fieldName 
		 */
		virtual void useFieldDiffusivityEverywhere(std::string _fieldName) { useFieldDiffusivityEverywhere(getFieldIndexByName(_fieldName)); }

		/**
		 * @brief Creates a watchable field at _fieldIndex which is stored in diffusivityFieldIndexToFieldMap 
		 * and registers a concentration field based on concentration name vector at _fieldIndex,
		 * diffusivityFieldSuffixStd and diffusivityFieldIndexToFieldMap
		 * 
		 * @param _fieldIndex 
		 */
		void initDiffusivityField(unsigned int _fieldIndex);

		/**
		 * @brief Finds _fieldIndex corresponding to _fieldName and creates a watchable field at _fieldIndex 
		 * which is stored in diffusivityFieldIndexToFieldMap and registers a concentration field 
		 * based on concentration name vector at _fieldIndex, diffusivityFieldSuffixStd and diffusivityFieldIndexToFieldMap
		 * The function is overridden by the derived class
		 * 
		 * @param _fieldName 
		 */
		virtual void initDiffusivityField(std::string _fieldName) { initDiffusivityField(getFieldIndexByName(_fieldName)); }

		/**
		 * @brief Gets _fieldName from _fieldIndex and sets setConcentrationFieldPtVal 
		 * at _pt using _fieldName and _val for diffusivityFieldIndexToFieldMap at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @param _val 
		 */
		void setDiffusivityFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);

		/**
		 * @brief Sets setConcentrationFieldPtVal 
		 * at _pt using _fieldName and _val for diffusivityFieldIndexToFieldMap at _fieldIndex
		 * The method will be overriden by the derived class
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @param _val 
		 */
		virtual void setDiffusivityFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setDiffusivityFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }

		/**
		 * @brief Gets the concentration field at _pt and returns the concentration at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @return float 
		 */
		float getConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt);

		/**
		 * @brief Gets _fieldName from _fieldIndex and then gets the concentration field at _pt 
		 * and returns the concentration at _fieldIndex
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @return float 
		 */
		float getConcentrationFieldPtVal(std::string _fieldName, Point3D _pt) { return getConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt); }

		/**
		 * @brief Gets the concentration field at _pt and sets the concentration at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @param _val 
		 */
		void setConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);

		/**
		 * @brief Gets the _fieldIndex from _fieldName and then gets the concentration field at _pt 
		 * and sets the concentration at _fieldIndex
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @param _val 
		 */
		void setConcentrationFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }

		/**
		 * @brief Calls useFixedFluxSurface for _fv which assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex) and 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedSurfaceFlux to surfaceFluxFunctionPtrs 
		 * at (_fieldIndex, _surfaceIndex) 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _outwardFluxVal 
		 * @param _fv 
		 */
		void useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _outwardFluxVal, ReactionDiffusionSolverFV *_fv);

		/**
		 * @brief Calls useFixedFluxSurface field Finite Volume at _pt which assigns _outwardFluxVal to bcVals at (_fieldIndex( obtained from _fieldName), 
		 * _surfaceIndex( obtained from _surfaceName)) and assigns a pointer of ReactionDiffusionSolverFV::fixedSurfaceFlux 
		 * to surfaceFluxFunctionPtrs at (_fieldIndex, _surfaceIndex). 
		 * This  method is overloaded by the derived class 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _outwardFluxVal 
		 * @param _pt 
		 */
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, Point3D _pt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_pt)); }
		
		/**
		 * @brief Calls useFixedFluxSurface field Finite Volume at _physPt which assigns _outwardFluxVal to bcVals at (_fieldIndex( obtained from _fieldName), 
		 * _surfaceIndex( obtained from _surfaceName)) and assigns a pointer of ReactionDiffusionSolverFV::fixedSurfaceFlux 
		 * to surfaceFluxFunctionPtrs at (_fieldIndex, _surfaceIndex). 
		 * This  method is overloaded by the derived class 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _outwardFluxVal 
		 * @param _physPt 
		 */
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, std::vector<float> _physPt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_physPt)); }
		
		/**
		 * @brief Calls useFixedConcentration for _fv which fills bcVals at (_fieldIndex, _surfaceIndex) and 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux to surfaceFluxFunctionPtrs 
		 * at (_fieldIndex, _surfaceIndex) 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _val 
		 * @param _fv 
		 */
		void useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _val, ReactionDiffusionSolverFV *_fv);
		
		/**
		 * @brief Calls useFixedConcentration field Finite Volume at _pt which fills bcVals at (_fieldIndex( obtained from _fieldName), 
		 * _surfaceIndex( obtained from _surfaceName)) and assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux 
		 * to surfaceFluxFunctionPtrs at (_fieldIndex, _surfaceIndex)
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _val 
		 * @param _pt 
		 */
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, Point3D _pt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_pt)); }
		
		/**
		 * @brief Calls useFixedConcentration field Finite Volume at _physPt which fills bcVals at (_fieldIndex( obtained from _fieldName), 
		 * _surfaceIndex( obtained from _surfaceName)) and assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux 
		 * to surfaceFluxFunctionPtrs at (_fieldIndex, _surfaceIndex)
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _val 
		 * @param _physPt 
		 */
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, std::vector<float> _physPt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_physPt)); }
		
		/**
		 * @brief Calls useFixedConcentration for _fv which assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex), 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux at (_fieldIndex, _surfaceIndex) to surfaceFluxFunctionPtrs and
		 * assigns diagonalFunctions and offDiagonalFunctions at _fieldIndex the function zeroMuParserFunction()
		 * _surfaceIndex ranges from 0 to length of surfaceFluxFunctionPtrs[_fieldIndex]
		 * 
		 * 
		 * 
		 * @param _fieldIndex 
		 * @param _val 
		 * @param _fv 
		 */
		void useFixedFVConcentration(unsigned int _fieldIndex, float _val, ReactionDiffusionSolverFV *_fv);
		
		/**
		 * @brief Calls useFixedConcentration for Finite Volume at _pt which assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex), 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux at (_fieldIndex, _surfaceIndex) to surfaceFluxFunctionPtrs and
		 * assigns diagonalFunctions and offDiagonalFunctions at _fieldIndex the function zeroMuParserFunction()
		 * _surfaceIndex ranges from 0 to length of surfaceFluxFunctionPtrs[_fieldIndex]
		 * _fieldIndex is derived from _fieldName
		 * 
		 * @param _fieldName 
		 * @param _val 
		 * @param _pt 
		 */
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, Point3D _pt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_pt)); }
		
		/**
		 * @brief Calls useFixedConcentration for Finite Volume at _physPt which assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex), 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux at (_fieldIndex, _surfaceIndex) to surfaceFluxFunctionPtrs and
		 * assigns diagonalFunctions and offDiagonalFunctions at _fieldIndex the function zeroMuParserFunction()
		 * _surfaceIndex ranges from 0 to length of surfaceFluxFunctionPtrs[_fieldIndex]
		 * _fieldIndex is derived from _fieldName
		 * 
		 * @param _fieldName 
		 * @param _val 
		 * @param _physPt 
		 */
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, std::vector<float> _physPt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_physPt)); }
		
		/**
		 * @brief use diffusive surface in a volume element for a field
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex
		 * @param _pt
		 */
		void useDiffusiveSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, Point3D _pt);
		
		/**
		 * @brief use diffusive surface in a volume element for a field
		 * 
		 * @param _fieldName 
		 * @param _surfaceName
		 * @param _pt
		 */
		void useDiffusiveSurface(std::string _fieldName, std::string _surfaceName, Point3D _pt) { useDiffusiveSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _pt); }
		
		/**
		 * @brief use diffusive surfaces in a volume element for a field
		 * 
		 * @param _fieldIndex 
		 * @param _pt
		 */
		void useDiffusiveSurfaces(unsigned int _fieldIndex, Point3D _pt);
		
		/**
		 * @brief use diffusive surfaces in a volume element for a field
		 * 
		 * @param _fieldName 
		 * @param _pt
		 */
		void useDiffusiveSurfaces(std::string _fieldName, Point3D _pt) { return useDiffusiveSurfaces(getFieldIndexByName(_fieldName), _pt); }
		
		/**
		 * @brief calls useDiffusiveSurfaces for each fieldFV in fieldFVs which sets surfaceFluxFunctionPtrs at 
		 * _fieldIndex using field Finite Volume pointers as pointer of ReactionDiffusionSolverFV::diffusiveSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 */
		void useDiffusiveSurfaces(unsigned int _fieldIndex);
		
		/**
		 * @brief calls useDiffusiveSurfaces for each fieldFV in fieldFVs which sets surfaceFluxFunctionPtrs at 
		 * _fieldIndex using field Finite Volume pointers as pointer of ReactionDiffusionSolverFV::diffusiveSurfaceFlux
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 */
		virtual void useDiffusiveSurfaces(std::string _fieldName) { useDiffusiveSurfaces(getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief  calls usePermeableSurface for the fieldFv at a point which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex
		 * @param _pt
		 */
		void usePermeableSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, Point3D _pt);
		
		/**
		 * @brief  calls usePermeableSurface for the fieldFv at a point which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldName 
		 * @param _surfaceName
		 * @param _pt
		 */
		void usePermeableSurface(std::string _fieldName, std::string _surfaceName, Point3D _pt) { usePermeableSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _pt); }
		
		/**
		 * @brief  calls usePermeableSurfaces for each fieldFv in fieldFVs which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 */
		void usePermeableSurfaces(unsigned int _fieldIndex) { usePermeableSurfaces(_fieldIndex, true); }
		
		/**
		 * @brief calls usePermeableSurfaces for each fieldFv in fieldFVs which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 * @param _activate 
		 */
		void usePermeableSurfaces(unsigned int _fieldIndex, bool _activate);
		
		/**
		 * @brief calls usePermeableSurfaces for each fieldFv in fieldFVs which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 * @param _activate 
		 */
		virtual void usePermeableSurfaces(std::string _fieldName, bool _activate) { usePermeableSurfaces(getFieldIndexByName(_fieldName), _activate); }

		// Cell interface
		/**
		 * @brief Initialises cell data for _cell and permeationCoefficients, permeableBiasCoefficients, diffusivityCoefficients and outwardFluxValues 
		 * 
		 * @param _cell 
		 * @param _numFields 
		 */
		void initializeCellData(const CellG *_cell, unsigned int _numFields);
		
		/**
		 * @brief calls initializeCellData for all cells in cellInventory
		 * 
		 * @param _numFields 
		 */
		void initializeCellData(unsigned int _numFields);
		
		/**
		 * @brief collects diffusivity coefficients for _cell and returns the value at _fieldIndex
		 * If the size of diffusivity coefficient vector is less than _fieldIndex, cell diffusivity
		 * coefficients are initialised on the fly  
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		
		/**
		 * @brief collects diffusivity coefficients for _cell and returns the value at _fieldIndex
		 * If the size of diffusivity coefficient vector is less than _fieldIndex, cell diffusivity
		 * coefficients are initialised on the fly 
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @return double 
		 */
		virtual double getCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { return getCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of diffusivity coefficients for _cell and adds _diffusivityCoefficient 
		 * at _fieldIndex in the diffusivity coefficients vector
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex, double _diffusivityCoefficient);
		
		/**
		 * @brief gets a vector of diffusivity coefficients for _cell and adds _diffusivityCoefficient 
		 * at _fieldIndex in the diffusivity coefficients vector
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName, double _diffusivityCoefficient) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		
		/**
		 * @brief gets a vector of diffusivity coefficients for _cell and adds _diffusivityCoefficient 
		 * at _fieldIndex in the diffusivity coefficients vector
		 * _fieldIndex is obtained from _fieldName
		 * Gets diffusivityCoefficient from constantDiffusionCoefficientsVecCellType at (_fieldIndex, _cell->type)
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 */
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		
		/**
		 * @brief gets a vector of diffusivity coefficients for _cell and adds _diffusivityCoefficient 
		 * at _fieldIndex in the diffusivity coefficients vector
		 * _fieldIndex is obtained from _fieldName
		 * Gets diffusivityCoefficient from constantDiffusionCoefficientsVecCellType at (_fieldIndex, _cell->type)
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 */
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief Sets diffusivity coefficients for each cell in cellInventory, for each _fieldIndex
		 * 
		 * @param _fieldIndex 
		 */
		void setCellDiffusivityCoefficients(unsigned int _fieldIndex);
		
		/**
		 * @brief  Sets diffusivity coefficients for each cell in cellInventory, for each _fieldIndex
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellDiffusivityCoefficients(std::string _fieldName) { setCellDiffusivityCoefficients(getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief Sets diffusivity coefficients for each cell in cellInventory, for each _fieldIndex
		 * 
		 */
		virtual void setCellDiffusivityCoefficients();
		
		/**
		 * @brief Gets the permeation and bias coefficient of a cell for a cell type and field
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @return vector of permeation coefficient and bias coefficient
		 */
		std::vector<double> getPermeableCoefficients(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		
		/**
		 * @brief Gets the permeation and bias coefficient of a cell for a cell type and field
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @return vector of permeation coefficient and bias coefficient
		 */
		std::vector<double> getPermeableCoefficients(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) {
			return getPermeableCoefficients(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief Sets bias coefficient and nBias coefficient for _cell and _nCell respectively
		 * Also sets permeationCoeffecient and then returns a vector containing permeationCoeffecient, 
		 * bias coefficient and nBias coefficient
		 * 
		 * @param _cell 
		 * @param _nCell 
		 * @param _fieldIndex 
		 * @return std::vector<double> 
		 */
		std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, unsigned int _fieldIndex);
		
		/**
		 * @brief Sets bias coefficient and nBias coefficient for _cell and _nCell respectively
		 * Also sets permeationCoeffecient and then returns a vector containing permeationCoeffecient, 
		 * bias coefficient and nBias coefficient
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _nCell 
		 * @param _fieldName 
		 * @return std::vector<double> 
		 */
		virtual std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, std::string _fieldName) {
			return getPermeableCoefficients(_cell, _nCell, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of permeation coefficients for _cell and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as _permeationCoefficient
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @param _permeationCoefficient 
		 */
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeationCoefficient);
		
		/**
		 * @brief gets a vector of permeation coefficients for _cell and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as _permeationCoefficient
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 * @param _permeationCoefficient 
		 */
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeationCoefficient) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeationCoefficient); }
		
		/**
		 * @brief gets a vector of permeation coefficients for _cell and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as the permeation coefficient correponding to the value in constantPermeationCoefficientsVecCellType
		 * at _fieldIndex and _cell.cellType
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 */
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		
		/**
		 * @brief gets a vector of permeation coefficients for _cell and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as the permeation coefficient correponding to the value in constantPermeationCoefficientsVecCellType
		 * at _fieldIndex and _cell.cellType
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 */
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of permeation coefficients for each cell in cellInventory  and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and each celltypeId in collection of celltypeIds
		 * as the permeation coefficient correponding to the value in constantPermeationCoefficientsVecCellType 
		 * at _fieldIndex and _cell.cellType
		 * 
		 * @param _fieldIndex 
		 */
		void setCellPermeationCoefficients(unsigned int _fieldIndex);
		
		/**
		 * @brief gets a vector of permeation coefficients for each cell in cellInventory  and then sets 
		 * the permeation coefficient corresponding to  _fieldIndex and each celltypeId in collection of celltypeIds
		 * as the permeation coefficient correponding to the value in constantPermeationCoefficientsVecCellType 
		 * at _fieldIndex and _cell.cellType 
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellPermeationCoefficients(std::string _fieldName) { setCellPermeationCoefficients(getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of permeation coefficients for each cell in cellInventory  and then sets 
		 * the permeation coefficient corresponding to  each _fieldIndex in range(0, numFields) and each celltypeId in collection of celltypeIds
		 * as the permeation coefficient correponding to the value in constantPermeationCoefficientsVecCellType 
		 * at _fieldIndex and _cell.cellType 
		 * 
		 */
		virtual void setCellPermeationCoefficients();
		
		/**
		 * @brief gets a vector of permeable bias coefficients for _cell and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as _permeableBiasCoefficient 
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @param _permeableBiasCoefficient 
		 */
		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeableBiasCoefficient);
		
		/**
		 * @brief gets a vector of permeable bias coefficients for _cell and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as _permeableBiasCoefficient 
		 * _fieldIndex is derived from _fieldName
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 * @param _permeableBiasCoefficient 
		 */
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeableBiasCoefficient) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeableBiasCoefficient); }
		
		/**
		 * @brief gets a vector of permeable bias coefficients for _cell and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as the permeable bias coefficient correponding to the value in constPermBiasCoeffsVecCellType
		 * at _fieldIndex and _cell.cellType
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 */
		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		
		/**
		 * @brief gets a vector of permeable bias coefficients for _cell and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and _nCellTypeID
		 * as the permeable bias coefficient correponding to the value in constPermBiasCoeffsVecCellType
		 * at _fieldIndex and _cell.cellType
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 */
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of permeable bias coefficients for each cell in cellInventory  and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and each celltypeId in collection of celltypeIds
		 * as the permeable bias coefficient correponding to the value in constPermBiasCoeffsVecCellType 
		 * at _fieldIndex and _cell.cellType
		 * 
		 * @param _fieldIndex 
		 */
		void setCellPermeableBiasCoefficients(unsigned int _fieldIndex);
		
		/**
		 * @brief gets a vector of permeable bias coefficients for each cell in cellInventory  and then sets 
		 * the permeable bias coefficient corresponding to  _fieldIndex and each celltypeId in collection of celltypeIds
		 * as the permeable bias coefficient correponding to the value in constPermBiasCoeffsVecCellType 
		 * at _fieldIndex and _cell.cellType
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellPermeableBiasCoefficients(std::string _fieldName) { setCellPermeableBiasCoefficients(getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief gets a vector of permeable bias coefficients for each cell in cellInventory  and then sets 
		 * the permeable bias coefficient corresponding to  each _fieldIndex in range(0, numFields) and each celltypeId in collection of celltypeIds
		 * as the permeable bias coefficient correponding to the value in constPermBiasCoeffsVecCellType 
		 * at _fieldIndex and _cell.cellType 
		 * 
		 */
		virtual void setCellPermeableBiasCoefficients();
		
		/**
		 * @brief Sets permeation and permeable bias coefficients
		 * 
		 */
		virtual void setCellPermeableCoefficients();
		
		/**
		 * @brief gets the outward flux value for _cell at _fieldIndex from a vector of outward flux values
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex);
		
		/**
		 * @brief gets the outward flux value for _cell at _fieldIndex from a vector of outward flux values
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @return double 
		 */
		virtual double getCellOutwardFlux(const CellG *_cell, std::string _fieldName) { return getCellOutwardFlux(_cell, getFieldIndexByName(_fieldName)); }
		
		/**
		 * @brief sets the outward flux value for _cell at _fieldIndex
		 * in a vector of outward flux values as _outwardFlux
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @param _outwardFlux 
		 */
		void setCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex, float _outwardFlux);
		
		/**
		 * @brief sets the outward flux value for _cell at _fieldIndex
		 * in a vector of outward flux values as _outwardFlux
		 * _fieldIndex is derived from _fieldName
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @param _outwardFlux 
		 */
		virtual void setCellOutwardFlux(const CellG * _cell, std::string _fieldName, float _outwardFlux) { setCellOutwardFlux(_cell, getFieldIndexByName(_fieldName), _outwardFlux); }


		// Solver functions

		/**
		 * @brief Checks if point is valid or not
		 * 
		 * @param pt 
		 * @return true 
		 * @return false 
		 */
		bool isValid(Point3D pt) { try { pt2ind(pt); return true; } catch (CC3DException) { return false; } }

		/**
		 * @brief Test whether a cell is in contact with a cell type
		 * 
		 * @param cell cell to test
		 * @param typeIndex cell type to test
		 */
		bool inContact(CellG *cell, const unsigned char &typeIndex);

		std::set<unsigned char> getNeighbors(CellG *cell);
		
		/**
		 * @brief Get the Lattice Point From Phys object
		 * 
		 * @param _physPt 
		 * @return Point3D 
		 */
		virtual Point3D getLatticePointFromPhys(std::vector<float> _physPt) { return Point3D((int)(_physPt[0] / lengthX), (int)(_physPt[1] / lengthY), (int)(_physPt[2] / lengthZ)); }
		
		/**
		 * @brief gets surface index corresponding to _surfaceName
		 * 
		 * @param _surfaceName 
		 * @return unsigned int 
		 */
		unsigned int getSurfaceIndexByName(std::string _surfaceName);
		
		/**
		 * @brief gets field index corresponding to _fieldName
		 * 
		 * @param _fieldName 
		 * @return unsigned int 
		 */
		unsigned int getFieldIndexByName(std::string _fieldName);
		
		/**
		 * @brief sets units time using _unitsTime
		 * 
		 * @param _unitsTime 
		 */
		void setUnitsTime(std::string _unitsTime = "s");
		
		/**
		 * @brief gets units time
		 * 
		 * @return std::string 
		 */
		std::string getUnitsTime() { return unitsTime; }
		
		/**
		 * @brief returns dimension index corresponding to _surfaceIndex
		 * 
		 * @param _surfaceIndex 
		 * @return unsigned int 
		 */
		unsigned int getIndexSurfToCoord(unsigned int _surfaceIndex) { return indexMapSurfToCoord[_surfaceIndex]; }
		
		/**
		 * @brief get surfaceNormSign corresponding to _surfaceIndex
		 * 
		 * @param _surfaceIndex 
		 * @return int 
		 */
		int getSurfaceNormSign(unsigned int _surfaceIndex) { return surfaceNormSign[_surfaceIndex]; }
		
		/**
		 * @brief gets length along x, y or z axis depending on the axis requested by _dimIndex
		 * 
		 * @param _dimIndex 
		 * @return float 
		 */
		virtual float getLength(int _dimIndex) { return std::vector<float>{ lengthX, lengthY, lengthZ }[_dimIndex]; }
		
		/**
		 * @brief gets length along x, y or z axis depending on the axis requested by _dimIndex
		 * corresponding to _surfaceIndex
		 * 
		 * @param _surfaceIndex 
		 * @return float 
		 */
		virtual float getLengthBySurfaceIndex(unsigned int _surfaceIndex) { return getLength(indexMapSurfToCoord[_surfaceIndex]); }
		
		/**
		 * @brief gets signed distance corresponding to _surfaceIndex
		 * 
		 * @param _surfaceIndex 
		 * @return float 
		 */
		float getSignedDistanceBySurfaceIndex(unsigned int _surfaceIndex) { return signedDistanceBySurfaceIndex[_surfaceIndex]; }
		
		/**
		 * @brief gets length of x,y and z axis
		 * 
		 * @return std::vector<float> 
		 */
		std::vector<float> getLengths() { return std::vector<float>{lengthX, lengthY, lengthZ}; }
		
		/**
		 * @brief sets length of x,y and z axis to same value using _length
		 * 
		 * @param _length 
		 */
		virtual void setLengths(float _length) { setLengths(_length, _length, _length); }
		
		/**
		 * @brief sets length of x,y and z using _lengthX, _lengthY and _lengthZ respectively
		 * 
		 * @param _lengthX 
		 * @param _lengthY 
		 * @param _lengthZ 
		 */
		virtual void setLengths(float _lengthX, float _lengthY, float _lengthZ);
		
		/**
		 * @brief gets surface area corresponding to _surfaceIndex
		 * 
		 * @param _surfaceIndex 
		 * @return float 
		 */
		virtual float getSurfaceArea(unsigned int _surfaceIndex) { return surfaceAreas[_surfaceIndex]; }
		
		/**
		 * @brief updates surface areas for lattices
		 * 
		 */
		virtual void updateSurfaceAreas();
		
		/**
		 * @brief Get the Time Step object
		 * 
		 * @return double 
		 */
		double getTimeStep() { return incTime; }
		
		/**
		 * @brief returns integration timestamp
		 * 
		 * @return double 
		 */
		double getIntegrationTimeStep() { return integrationTimeStep; }
		
		/**
		 * @brief returns physical time
		 * 
		 * @return double 
		 */
		double getPhysicalTime() { return physTime; }
		
		/**
		 * @brief gets the field finite volume corresponding to the index obtained from _pt
		 * 
		 * @param _pt 
		 * @return ReactionDiffusionSolverFV* 
		 */
		virtual ReactionDiffusionSolverFV * getFieldFV(const Point3D &_pt) { return getFieldFV(pt2ind(_pt)); }
		
		/**
		 * @brief gets the field finite volume corresponding to the index _ind
		 * 
		 * @param _ind 
		 * @return ReactionDiffusionSolverFV* 
		 */
		virtual ReactionDiffusionSolverFV * getFieldFV(unsigned int _ind) { return fieldFVs[_ind]; }
		
		/**
		 * @brief generates lattice point from _physPt and then gets field finite volume from the lattice point 
		 * 
		 * @param _physPt 
		 * @return ReactionDiffusionSolverFV* 
		 */
		virtual ReactionDiffusionSolverFV * getFieldFV(std::vector<float> _physPt) { return getFieldFV(getLatticePointFromPhys(_physPt)); }
		
		/**
		 * @brief sets field finite volume for reaction diffusion solver _fv using index from point _pt
		 * 
		 * @param _pt 
		 * @param _fv 
		 */
		void setFieldFV(Point3D _pt, ReactionDiffusionSolverFV *_fv) { setFieldFV(pt2ind(_pt), _fv); }
		
		/**
		 * @brief sets field finite volume for reaction diffusion solver _fv using index _ind
		 * 
		 * @param _ind 
		 * @param _fv 
		 */
		void setFieldFV(unsigned int _ind, ReactionDiffusionSolverFV *_fv) { fieldFVs[_ind] = _fv; }
		
		/**
		 * @brief gets neihbouring finite volumes for _fv
		 * 
		 * @param _fv 
		 * @return std::map<unsigned int, ReactionDiffusionSolverFV *> 
		 */
		std::map<unsigned int, ReactionDiffusionSolverFV *> getFVNeighborFVs(ReactionDiffusionSolverFV *_fv);
		
		/**
		 * @brief gets Point3D object corresponding to index _ind
		 * 
		 * @param _ind 
		 * @return Point3D 
		 */
		Point3D ind2pt(unsigned int _ind);
		
		/**
		 * @brief gets index corresponding to Point3D object _pt
		 * _fieldDim contains length of dimensions across x, y and z axis
		 * 
		 * @param _pt 
		 * @param _fieldDim 
		 * @return unsigned int 
		 */
		unsigned int pt2ind(const Point3D &_pt, Dim3D _fieldDim);
		
		/**
		 * @brief gets index corresponding to Point3D object _pt
		 * 
		 * @param _pt 
		 * @return unsigned int 
		 */
		unsigned int pt2ind(const Point3D &_pt) { return pt2ind(_pt, fieldDim); }
		
		/**
		 * @brief returns field dimensions object
		 * 
		 * @return Dim3D 
		 */
		Dim3D getFieldDim() { return fieldDim; }

		// Solver interface
		
		/**
		 * @brief sets fieldExpressionStringsMergedDiag at _fieldIndex as _expr
		 * 
		 * @param _fieldIndex 
		 * @param _expr 
		 */
		void setFieldExpressionMultiplier(unsigned int _fieldIndex, std::string _expr);
		
		/**
		 * @brief sets fieldExpressionStringsMergedDiag at _fieldIndex as _expr
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 * @param _expr 
		 */
		virtual void setFieldExpressionMultiplier(std::string _fieldName, std::string _expr) { setFieldExpressionMultiplier(getFieldIndexByName(_fieldName), _expr); }
		
		/**
		 * @brief sets fieldExpressionStringsMergedOffDiag at _fieldIndex as _expr
		 * 
		 * @param _fieldIndex 
		 * @param _expr 
		 */
		void setFieldExpressionIndependent(unsigned int _fieldIndex, std::string _expr);
		
		/**
		 * @brief  sets fieldExpressionStringsMergedOffDiag at _fieldIndex as _expr
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 * @param _expr 
		 */
		virtual void setFieldExpressionIndependent(std::string _fieldName, std::string _expr) { setFieldExpressionIndependent(getFieldIndexByName(_fieldName), _expr); }
		
		/**
		 * @brief gets field symbol corresponding to _fieldIndex in fieldSymbolsVec
		 * 
		 * @param _fieldIndex 
		 * @return std::string 
		 */
		std::string getFieldSymbol(unsigned int _fieldIndex) { return fieldSymbolsVec[_fieldIndex]; }
		
		/**
		 * @brief gets field symbol corresponding to _fieldIndex in fieldSymbolsVec
		 * _fieldIndex is obtained from _fieldName
		 * 
		 * @param _fieldName 
		 * @return std::string 
		 */
		virtual std::string getFieldSymbol(std::string _fieldName) { return fieldSymbolsVec[getFieldIndexByName(_fieldName)]; }

	};

	/////////////////////////////////////////////////////////////// Field wrap ///////////////////////////////////////////////////////////////
	template <class T>
	class PDESOLVERS_EXPORT RDFVMField3DWrap: public Array3DBordersField3DAdapter<T>
	{
	private:
		ReactionDiffusionSolverFVM *solver;
		std::string fieldName;
	public:
		RDFVMField3DWrap(ReactionDiffusionSolverFVM *_solver, std::string _fieldName) : solver(_solver), fieldName(_fieldName) {};
		virtual ~RDFVMField3DWrap() {};

		void set(const Point3D &pt, const T value);
		T get(const Point3D &pt) const;
		void setByIndex(long _offset, const T value);
		T getByIndex(long _offset) const;
		T operator[](const Point3D &pt) const { return get(pt); }
		Dim3D getDim() const;
		bool isValid(const Point3D &pt) const;
		void setDim(const Dim3D theDim) {}
		void resizeAndShift(const Dim3D theDim, const Dim3D shiftVec) {}
		void clearSecData() {}
	};

	////////////////////////////////////////////////////////////// Finite volume /////////////////////////////////////////////////////////////
	// Solver algorithm 1: fixed time step ( using solve() )		-> each FV calculates and stores solution increment without time step
	// Solver algorithm 2: stable time step ( using solveStable() )	-> each FV calculates and stores solution increment without time step and returns maximum stable time step
	//																-> solver accepts minimum of all FV maximum stable time steps
	// Update algorithm ( using update() )							-> solver passes time increment to each FV
	//																-> each FV updates according to stored solution increment and time increment passed from solver

	// ReactionDiffusionSolverFVM Finite volume class
	class PDESOLVERS_EXPORT ReactionDiffusionSolverFV
	{
		typedef std::vector<double>(ReactionDiffusionSolverFV::*FluxFunction)(unsigned int, unsigned int, ReactionDiffusionSolverFV *);
		typedef double(ReactionDiffusionSolverFV::*DiffusivityFunction)(unsigned int);

	private:
		ReactionDiffusionSolverFVM *solver;
		Point3D coords;
		std::vector<double> concentrationVecAux;
		std::vector<double> concentrationVecOld;
		std::vector<double> secrRateStorage;
		std::map<unsigned int, ReactionDiffusionSolverFV *> neighborFVs;

		double physTime;
		bool stabilityMode;
		bool usingCellInterfaceFlux;

		// Dim 1: field
		// Dim 2: diffusion coefficient, ...
		std::vector<std::vector<double> > auxVars;
		// Dim 1: field
		// Dim 2, fixed concentration: concentration value
		// Dim 2, fixed flux: outward flux by surface index
		std::vector<std::vector<double> > bcVals;

		std::vector<std::vector<FluxFunction> > surfaceFluxFunctionPtrs;
		std::vector<mu::Parser> diagonalFunctions;
		std::vector<mu::Parser> offDiagonalFunctions;

		std::vector<DiffusivityFunction> fieldDiffusivityFunctionPtrs;

		double diagonalFunctionEval(unsigned int _fieldIndex);
		double offDiagonalFunctionEval(unsigned int _fieldIndex);

		double returnZero() { return 0.0; }
		double returnZero(unsigned int _fieldIndex) { return 0.0; }
		std::vector<double> returnZero(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv) { return std::vector<double>{0.0, 0.0, 0.0}; }

		void setFieldDiffusivityFunction(unsigned int _fieldIndex, DiffusivityFunction _fcn) { fieldDiffusivityFunctionPtrs[_fieldIndex] = _fcn; }
		double getConstantDiffusivity(unsigned int _fieldIndex) { return auxVars[_fieldIndex][0]; }
		double getConstantDiffusivityById(unsigned int _fieldIndex);
		double getFieldDiffusivityField(unsigned int _fieldIndex);
		double getFieldDiffusivityInMedium(unsigned int _fieldIndex);

		double cellInterfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, CellG *cell, CellG *nCell);
		double cellInterfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> diffusiveSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> permeableSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedFVConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		
		double secreteSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData);
		double secreteOnContactSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData);
		double secreteConstantConcentrationSingleField(const unsigned int &fieldIndex, const unsigned char &typeIndex, const SecretionData &secrData);

	public:


		ReactionDiffusionSolverFV() : usingCellInterfaceFlux{false} {};
		ReactionDiffusionSolverFV(ReactionDiffusionSolverFVM *_solver, Point3D _coords, int _numFields);
		virtual ~ReactionDiffusionSolverFV() {};

		/**
		 * @brief returns (x,y,z) coordinates 
		 * 
		 * @return const Point3D 
		 */
		const Point3D getCoords() { return coords; }

		/**
		 * @brief Initialises neighbour finite volumes, bcVals and surface flux function pointers
		 * 
		 */
		void initialize();

		/**
		 * @brief Performs secretion and stores results internally for subsequent call to update method
		 * 
		 * @param secrFieldVec vector of secretion data, ordered by field
		 */
		void secrete(const std::vector<SecretionData> &secrFieldVec);
		
		/**
		 * @brief Evaluates diagonal functions, multiplies it with old concentration and adds the result
		 * to the result from evaluation of off diagonal functions for every element in old concentration
		 * The final result is accumulated in concentrationVecAux
		 * 
		 */
		void solve();
		
		/**
		 * @brief Evaluates diagonal functions, multiplies it with old concentration and adds the result
		 * to the result from evaluation of off diagonal functions for every element in old concentration
		 * and returns _incTime
		 * The final result is accumulated in concentrationVecAux
		 * 
		 * @return double 
		 */
		double solveStable();
		
		/**
		 * @brief Updates values in old concentration vector and adds _incTime to physical time
		 * 
		 * @param _incTime 
		 */
		void update(double _incTime);

		/**
		 * @brief clears the vector of neighbour finite volumes
		 * 
		 */
		void clearNeighbors() { neighborFVs.clear(); }

		/**
		 * @brief Adds a neighbour finite volume which is a pair of ReactionDiffusionSolver element and a surface index
		 * given by _fv and _surfaceIndex respectively
		 * 
		 * @param _fv 
		 * @param _surfaceIndex 
		 */
		void addNeighbor(ReactionDiffusionSolverFV *_fv, unsigned int _surfaceIndex) { neighborFVs.insert(make_pair(_surfaceIndex, _fv)); }
		
		/**
		 * @brief returns a map of neighbour finite volumes consisting of pairs of ReactionDiffusionSolver element and a surface index
		 * 
		 * @return std::map<unsigned int, ReactionDiffusionSolverFV *> 
		 */
		std::map<unsigned int, ReactionDiffusionSolverFV *> getNeighbors() { return neighborFVs; }

		/**
		 * @brief  Sets pointers to getConstantDiffusivity function
		 * for fieldDiffusivityFunctionPtrs vector and sets constant diffusivity
		 * given by solver's getConstantFieldDiffusivity function for auxVars at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex);
		
		/**
		 * @brief  Sets pointers to getConstantDiffusivity function
		 * for fieldDiffusivityFunctionPtrs vector and sets constant diffusivity
		 * given by _constantDiff for auxVars at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @param _constantDiff 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex, double _constantDiff);
		
		/**
		 * @brief  Sets pointers to getConstantDiffusivity function
		 * for getConstantDiffusivityById vector and sets constant diffusivity
		 * given by solver's getConstantFieldDiffusivity function for auxVars at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivityById(unsigned int _fieldIndex);
		
		/**
		 * @brief Sets pointers to getConstantDiffusivity function
		 * for getConstantDiffusivityById vector and sets constant diffusivity
		 * given by _constantDiff for auxVars at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @param _constantDiff 
		 */
		void useConstantDiffusivityById(unsigned int _fieldIndex, double _constantDiff);
		
		/**
		 * @brief sets fieldDiffusivityFunctionPtrs at _fieldIndex to a pointer to getFieldDiffusivityField function in ReactionDiffusionSolverFV
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityEverywhere(unsigned int _fieldIndex) { fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getFieldDiffusivityField; }
		
		/**
		 * @brief sets fieldDiffusivityFunctionPtrs at _fieldIndex to a pointer to getFieldDiffusivityInMedium function in ReactionDiffusionSolverFV
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityInMedium(unsigned int _fieldIndex) { fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getFieldDiffusivityInMedium; }
		
		/**
		 * @brief calls useDiffusiveSurface for each fieldFV in fieldFVs which sets surfaceFluxFunctionPtrs at 
		 * _fieldIndex using field Finite Volume pointers as pointer of ReactionDiffusionSolverFV::diffusiveSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex
		 */
		void useDiffusiveSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex);
		
		/**
		 * @brief calls useDiffusiveSurfaces for each fieldFV in fieldFVs which sets surfaceFluxFunctionPtrs at 
		 * _fieldIndex using field Finite Volume pointers as pointer of ReactionDiffusionSolverFV::diffusiveSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 */
		void useDiffusiveSurfaces(unsigned int _fieldIndex);
		
		/**
		 * @brief calls usePermeableSurface for each fieldFv in fieldFVs which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _activate 
		 */
		void usePermeableSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, bool _activate = true);
		
		/**
		 * @brief calls usePermeableSurfaces for each fieldFv in fieldFVs which sets surfaceFluxFunctionPtrs
		 * at _fieldIndex using field Finite Volume pointers as a pointer of ReactionDiffusionSolverFV::permeableSurfaceFlux
		 * 
		 * @param _fieldIndex 
		 * @param _activate 
		 */
		void usePermeableSurfaces(unsigned int _fieldIndex, bool _activate = true);
		
		/**
		 * @brief assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex) and 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedSurfaceFlux to surfaceFluxFunctionPtrs 
		 * at (_fieldIndex, _surfaceIndex) 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _fluxVal 
		 */
		void useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _fluxVal);
		
		/**
		 * @brief fills bcVals at (_fieldIndex, _surfaceIndex) and assigns a pointer of 
		 * ReactionDiffusionSolverFV::fixedConcentrationFlux to surfaceFluxFunctionPtrs 
		 * at (_fieldIndex, _surfaceIndex) 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _val 
		 */
		void useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _val);
		
		/**
		 * @brief Assigns _outwardFluxVal to bcVals at (_fieldIndex, _surfaceIndex), 
		 * assigns a pointer of ReactionDiffusionSolverFV::fixedConcentrationFlux at (_fieldIndex, _surfaceIndex) to surfaceFluxFunctionPtrs and
		 * assigns diagonalFunctions and offDiagonalFunctions at _fieldIndex the function zeroMuParserFunction()
		 * _surfaceIndex ranges from 0 to length of surfaceFluxFunctionPtrs[_fieldIndex]
		 * 
		 * @param _fieldIndex 
		 * @param _val 
		 */
		void useFixedFVConcentration(unsigned int _fieldIndex, double _val);

		/**
		 * @brief Set flag for whether to use cell interface fluxes
		 */
		void useCellInterfaceFlux(const bool &_usingCellInterfaceFlux) { usingCellInterfaceFlux = _usingCellInterfaceFlux; }

		/**
		 * @brief registers field symbol by calling muParser::DefineVar that takes in  _fieldSymbol and old 
		 * concentration at _fieldIndex as input; on elements of diagonalFunctions and offDiagonalFunctions 
		 * at each index in range(0, length of diagonalFunctions)
		 * 
		 * @param _fieldIndex 
		 * @param _fieldSymbol 
		 */
		void registerFieldSymbol(unsigned int _fieldIndex, std::string _fieldSymbol);
		
		/**
		 * @brief sets expresion in diagonalFunctions at index _fieldIndex using _expr
		 * 
		 * @param _fieldIndex 
		 * @param _expr 
		 */
		void setDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr);
		
		/**
		 * @brief sets expresion in offDiagonalFunctions at index _fieldIndex using _expr
		 * 
		 * @param _fieldIndex 
		 * @param _expr 
		 */
		void setOffDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr);

		/**
		 * @brief defines an instance of mu parser and sets it's expression as '0.0' and returns that instance
		 * 
		 * @return mu::Parser 
		 */
		mu::Parser zeroMuParserFunction();
		
		/**
		 * @brief creates a zeroMuParserFunction and defines constants x,y,z,t corresponding to x,y,z coordinates and physical time respectively
		 * 
		 * @return mu::Parser 
		 */
		mu::Parser templateParserFunction();
		
		/**
		 * @brief sets old concentrations using _concentrationVec
		 * 
		 * @param _concentrationVec 
		 */
		void setConcentrationVec(std::vector<double> _concentrationVec);
		
		/**
		 * @brief sets value in concentrationVecOld at _fieldIndex using _concentrationVal
		 * 
		 * @param _fieldIndex 
		 * @param _concentrationVal 
		 */
		void setConcentration(unsigned int _fieldIndex, double _concentrationVal) { concentrationVecOld[_fieldIndex] = _concentrationVal; }
		
		/**
		 * @brief Adds contents of _concentrationVecInc in concentrationVecOld and then returns max(concentrationVecOld,0.0)
		 * 
		 * @param _concentrationVecInc 
		 */
		void addConcentrationVecIncrement(std::vector<double> _concentrationVecInc);
		
		/**
		 * @brief returns a vector of old concentrations
		 * 
		 * @return std::vector<double> 
		 */
		std::vector<double> getConcentrationVec() { return concentrationVecOld; }
		
		/**
		 * @brief returns concentration at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getConcentration(unsigned int _fieldIndex) { return concentrationVecOld[_fieldIndex]; }
		
		/**
		 * @brief returns old concentration at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getConcentrationOld(unsigned int _fieldIndex) { return concentrationVecOld[_fieldIndex]; }

		/**
		 * @brief returns field diffusivity at _fieldIndex
		 * 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getFieldDiffusivity(unsigned int _fieldIndex);

		/**
		 * @brief gets the cell-dependent outward flux value for a field
		 * 
		 * @param _fieldIndex 
		 */
		double getCellOutwardFlux(unsigned int _fieldIndex);

		/**
		 * @brief gets the cell-dependent outward flux values for all field
		 * 
		 */
		std::vector<double> getCellOutwardFluxes();

	};

	///////////////////////////////////////////////////////////// Cell parameters ////////////////////////////////////////////////////////////
	class PDESOLVERS_EXPORT ReactionDiffusionSolverFVMCellData{
	public:

		ReactionDiffusionSolverFVMCellData() {};
		~ReactionDiffusionSolverFVMCellData() {};

		std::vector<std::vector<double> > permeationCoefficients;
		std::vector<std::vector<double> > permeableBiasCoefficients;
		std::vector<double> diffusivityCoefficients;
		std::vector<double> outwardFluxValues;

	};

};
#endif