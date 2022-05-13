#ifndef COMPUCELL3DREACTIONDIFFUSIONSOLVERFVM_H
#define COMPUCELL3DREACTIONDIFFUSIONSOLVERFVM_H

#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "DiffusableVector.h"

#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTracker.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>
// #include <CompuCell3D/plugins/ECMaterials/ECMaterialsPlugin.h>

#include "DiffSecrData.h"
#include "BoundaryConditionSpecifier.h"

#include <CompuCell3D/CC3DEvents.h>

#include <string>

#include <vector>
//#include <concurrent_vector.h>
#include <set>
#include <map>
#include <iostream>
#include <muParser/muParser.h>
//#include <ppl.h>

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

	class BoundaryPixelTrackerData;
	class BoundaryPixelTrackerPlugin;
	class PixelTrackerData;
	class PixelTrackerPlugin;
	class CellTypePlugin;
	// class ECMaterialsPlugin;

	class DiffusionData;
	class SecretionDataFlex;

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
	class PDESOLVERS_EXPORT ReactionDiffusionSolverFVM :public DiffusableVector<float>, public CellGChangeWatcher
	{

	public:
		// typedef void (ReactionDiffusionSolverFVM::*diffSecrFcnPtr_t)(void);
		typedef void(ReactionDiffusionSolverFVM::*DiffusivityModeInitializer)(unsigned int);
		typedef void(ReactionDiffusionSolverFVM::*FluxConditionInitializer)(unsigned int);
		typedef void(ReactionDiffusionSolverFVM::*Field3DChangeFcn)(const Point3D &, CellG *, CellG *);
		typedef void(ReactionDiffusionSolverFVM::*Field3DAdditionalPtFcn)(const Point3D &);

	private:
		BasicClassAccessor<ReactionDiffusionSolverFVMCellData> ReactionDiffusionSolverFVMCellDataAccessor;

		BoundaryPixelTrackerPlugin *boundaryTrackerPlugin;
		PixelTrackerPlugin *pixelTrackerPlugin;
		CellTypePlugin *cellTypePlugin;
		// ECMaterialsPlugin *ecMaterialsPlugin;

		Field3DChangeFcn field3DChangeFcnPtr;
		Field3DAdditionalPtFcn field3DAdditionalPtFcnPtr;

		void field3DAdditionalPtPreStartup(const Point3D &ptAdd);
		void field3DAdditionalPtPostStartup(const Point3D &ptAdd);
		void field3DChangePreStartup(const Point3D &pt, CellG *newCell, CellG *oldCell) {}
		void field3DChangePostStartup(const Point3D &pt, CellG *newCell, CellG *oldCell);

		//replace with std::vector
		//concurrency::concurrent_vector<ReactionDiffusionSolverFV *> * fieldFVs;
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

		std::vector<FluxConditionInitializer> fluxConditionInitializerPtrs;

		std::vector<std::vector<tuple<std::string, float> > > basicBCData;
		std::vector<bool> periodicBoundaryCheckVector;

		unsigned int numCellTypes;
		std::map<std::string, unsigned int> cellTypeNameToIndexMap;
		std::vector<std::vector<double> > constantDiffusionCoefficientsVecCellType;
		std::vector<DiffusivityModeInitializer> diffusivityModeInitializerPtrs;
		void returnVoid(unsigned int _fieldIndex) {}

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

		CellInventory *cellInventory;
		CellInventory::cellInventoryIterator cell_itr;

		WatchableField3D<CellG *> *cellFieldG;

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

		//replace with std::vector
		//concurrency::concurrent_vector<double> *fvMaxStableTimeSteps;
		std::vector<double> *fvMaxStableTimeSteps;

		bool simpleMassConservation;
		Point3D flipSourcePt;

		unsigned int numberOfFields;
		Dim3D fieldDim;

		bool usingECMaterials;

	public:

		ReactionDiffusionSolverFVM();

		virtual ~ReactionDiffusionSolverFVM();

		BasicClassAccessor<ReactionDiffusionSolverFVMCellData> * getReactionDiffusionSolverFVMCellDataAccessorPtr() { return & ReactionDiffusionSolverFVMCellDataAccessor; }

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

		// Cell field watcher interface
		virtual void field3DAdditionalPt(const Point3D &ptAdd);
		virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

		std::vector<double> concentrationVecIncSMC;
		std::vector<double> concentrationVecCopiesMedium;
		std::vector<double> massConsCorrectionFactorsMedium;
		PixelTrackerPlugin *getPixelTrackerPlugin() { return pixelTrackerPlugin; }

		// Diffusable Vector interface
		Array3DBordersField3DAdapter<float> * getConcentrationField(const std::string & name) { return (Array3DBordersField3DAdapter<float> *) concentrationFieldVector[getFieldIndexByName(name)]; }
		void allocateDiffusableFieldVector(unsigned int numberOfFields, Dim3D fieldDim) { 
			boundaryStrategy = BoundaryStrategy::getInstance();
			maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
			concentrationFieldNameVector.assign(numberOfFields, std::string());
		}
		std::vector<std::string> getConcentrationFieldNameVector() { return concentrationFieldNameVector; }

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
		
		/**
		 * @brief Returns a vector containing total cell concentration 
		 * for each field index based on cell concentration at field finite volume 
		 * at each pixel var index
		 * 
		 * @param _cell 
		 * @return std::vector<double> 
		 */
		std::vector<double> totalCellConcentration(const CellG *_cell);
		
		/**
		 * @brief Returns total cell concentration across all fields
		 * 
		 * @return std::vector<double> 
		 */
		std::vector<double> totalMediumConcentration();
		
		/**
		 * @brief Updates concentrationVecCopies and massConsCorrectionFactors for cells in cellInventory
		 * 
		 */
		void updateTotalCellConcentrations();
		
		/**
		 * @brief Updates total mass in medium and total mass in cells
		 * 
		 */
		void updateTotalConcentrations();
		
		/**
		 * @brief Get the Cell Pixel Vec object
		 * 
		 * @param _cell 
		 * @return std::vector<Point3D> 
		 */
		std::vector<Point3D> getCellPixelVec(const CellG *_cell);
		
		/**
		 * @brief Get the Medium Pixel Vec object
		 * 
		 * @return std::vector<Point3D> 
		 */
		std::vector<Point3D> getMediumPixelVec();
		
		/**
		 * @brief Adds up concentrations for each index in _pixelVec for each field
		 * and returns a vector containing total concentrations per field  
		 * 
		 * @param _pixelVec 
		 * @return std::vector<double> 
		 */
		std::vector<double> totalPixelSetConcentration(std::vector<Point3D> _pixelVec);
		
		// FV interface
		/**
		 * @brief Get the Coords Of F V object
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
		 * @brief Get the Constant Field Diffusivity object
		 * 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getConstantFieldDiffusivity(unsigned int _fieldIndex) { return constantDiffusionCoefficientsVec[_fieldIndex]; }
		// float getECMaterialDiffusivity(unsigned int _fieldIndex, Point3D coords) { return ecMaterialsPlugin->getLocalDiffusivity(coords, concentrationFieldNameVector[_fieldIndex]); }
		
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param coords 
		 * @return float 
		 */
		float getECMaterialDiffusivity(unsigned int _fieldIndex, Point3D coords) { return 0.0; } // To be implemented with commented declaration/definition
		/**
		 * @brief Get the Diffusivity Field Pt Val object
		 * 
		 * @param _fieldIndex 
		 * @param pt 
		 * @return float 
		 */
		float getDiffusivityFieldPtVal(unsigned int _fieldIndex, const Point3D &pt) { return diffusivityFieldIndexToFieldMap[_fieldIndex]->get(pt); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex) { useConstantDiffusivity(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void useConstantDiffusivity(unsigned int _fieldIndex, double _diffusivityCoefficient);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void useConstantDiffusivity(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivity(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void useConstantDiffusivityByType(unsigned int _fieldIndex) { useConstantDiffusivityByType(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void useConstantDiffusivityByType(unsigned int _fieldIndex, double _diffusivityCoefficient);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void useConstantDiffusivityByType(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivityByType(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityInMedium(unsigned int _fieldIndex);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 */
		virtual void useFieldDiffusivityInMedium(std::string _fieldName) { useFieldDiffusivityInMedium(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void useFieldDiffusivityEverywhere(unsigned int _fieldIndex);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 */
		virtual void useFieldDiffusivityEverywhere(std::string _fieldName) { useFieldDiffusivityEverywhere(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void initDiffusivityField(unsigned int _fieldIndex);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 */
		virtual void initDiffusivityField(std::string _fieldName) { initDiffusivityField(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Diffusivity Field Pt Val object
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @param _val 
		 */
		void setDiffusivityFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);
		/**
		 * @brief Set the Diffusivity Field Pt Val object
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @param _val 
		 */
		virtual void setDiffusivityFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setDiffusivityFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }
		/**
		 * @brief Get the Concentration Field Pt Val object
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @return float 
		 */
		float getConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt);
		/**
		 * @brief Get the Concentration Field Pt Val object
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @return float 
		 */
		float getConcentrationFieldPtVal(std::string _fieldName, Point3D _pt) { return getConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt); }
		/**
		 * @brief Set the Concentration Field Pt Val object
		 * 
		 * @param _fieldIndex 
		 * @param _pt 
		 * @param _val 
		 */
		void setConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);
		/**
		 * @brief Set the Concentration Field Pt Val object
		 * 
		 * @param _fieldName 
		 * @param _pt 
		 * @param _val 
		 */
		void setConcentrationFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _outwardFluxVal 
		 * @param _fv 
		 */
		void useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _outwardFluxVal, ReactionDiffusionSolverFV *_fv);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _outwardFluxVal 
		 * @param _pt 
		 */
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, Point3D _pt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_pt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _outwardFluxVal 
		 * @param _physPt 
		 */
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, std::vector<float> _physPt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_physPt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _surfaceIndex 
		 * @param _val 
		 * @param _fv 
		 */
		void useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _val, ReactionDiffusionSolverFV *_fv);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _val 
		 * @param _pt 
		 */
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, Point3D _pt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_pt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _surfaceName 
		 * @param _val 
		 * @param _physPt 
		 */
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, std::vector<float> _physPt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_physPt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _val 
		 * @param _fv 
		 */
		void useFixedFVConcentration(unsigned int _fieldIndex, float _val, ReactionDiffusionSolverFV *_fv);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _val 
		 * @param _pt 
		 */
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, Point3D _pt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_pt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _val 
		 * @param _physPt 
		 */
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, std::vector<float> _physPt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_physPt)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void useDiffusiveSurfaces(unsigned int _fieldIndex);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 */
		virtual void useDiffusiveSurfaces(std::string _fieldName) { useDiffusiveSurfaces(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 */
		void usePermeableSurfaces(unsigned int _fieldIndex) { usePermeableSurfaces(_fieldIndex, true); }
		/**
		 * @brief 
		 * 
		 * @param _fieldIndex 
		 * @param _activate 
		 */
		void usePermeableSurfaces(unsigned int _fieldIndex, bool _activate);
		/**
		 * @brief 
		 * 
		 * @param _fieldName 
		 * @param _activate 
		 */
		virtual void usePermeableSurfaces(std::string _fieldName, bool _activate) { usePermeableSurfaces(getFieldIndexByName(_fieldName), _activate); }

		// Cell interface
		/**
		 * @brief 
		 * 
		 * @param _cell 
		 * @param _numFields 
		 */
		void initializeCellData(const CellG *_cell, unsigned int _numFields);
		/**
		 * @brief 
		 * 
		 * @param _numFields 
		 */
		void initializeCellData(unsigned int _numFields);
		/**
		 * @brief Get the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		/**
		 * @brief Get the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @return double 
		 */
		virtual double getCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { return getCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @param _diffusivityCoefficient 
		 */
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex, double _diffusivityCoefficient);
		/**
		 * @brief Set the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @param _diffusivityCoefficient 
		 */
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName, double _diffusivityCoefficient) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		/**
		 * @brief Set the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 */
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Diffusivity Coefficient object
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 */
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Diffusivity Coefficients object
		 * 
		 * @param _fieldIndex 
		 */
		void setCellDiffusivityCoefficients(unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Diffusivity Coefficients object
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellDiffusivityCoefficients(std::string _fieldName) { setCellDiffusivityCoefficients(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Diffusivity Coefficients object
		 * 
		 */
		virtual void setCellDiffusivityCoefficients();
		/**
		 * @brief Get the Permeable Coefficients object
		 * 
		 * @param _cell 
		 * @param _nCell 
		 * @param _fieldIndex 
		 * @return std::vector<double> 
		 */
		std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, unsigned int _fieldIndex);
		/**
		 * @brief Get the Permeable Coefficients object
		 * 
		 * @param _cell 
		 * @param _nCell 
		 * @param _fieldName 
		 * @return std::vector<double> 
		 */
		virtual std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, std::string _fieldName) {
			return getPermeableCoefficients(_cell, _nCell, getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Permeation Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @param _permeationCoefficient 
		 */
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeationCoefficient);
		/**
		 * @brief Set the Cell Permeation Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 * @param _permeationCoefficient 
		 */
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeationCoefficient) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeationCoefficient); }
		/**
		 * @brief Set the Cell Permeation Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 */
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Permeation Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 */
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Permeation Coefficients object
		 * 
		 * @param _fieldIndex 
		 */
		void setCellPermeationCoefficients(unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Permeation Coefficients object
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellPermeationCoefficients(std::string _fieldName) { setCellPermeationCoefficients(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Permeation Coefficients object
		 * 
		 */
		virtual void setCellPermeationCoefficients();
		/**
		 * @brief Set the Cell Permeable Bias Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 * @param _permeableBiasCoefficient 
		 */
		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeableBiasCoefficient);
		/**
		 * @brief Set the Cell Permeable Bias Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 * @param _permeableBiasCoefficient 
		 */
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeableBiasCoefficient) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeableBiasCoefficient); }
		/**
		 * @brief Set the Cell Permeable Bias Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldIndex 
		 */
		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Permeable Bias Coefficient object
		 * 
		 * @param _cell 
		 * @param _nCellTypeId 
		 * @param _fieldName 
		 */
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Permeable Bias Coefficients object
		 * 
		 * @param _fieldIndex 
		 */
		void setCellPermeableBiasCoefficients(unsigned int _fieldIndex);
		/**
		 * @brief Set the Cell Permeable Bias Coefficients object
		 * 
		 * @param _fieldName 
		 */
		virtual void setCellPermeableBiasCoefficients(std::string _fieldName) { setCellPermeableBiasCoefficients(getFieldIndexByName(_fieldName)); }
		/**
		 * @brief Set the Cell Permeable Bias Coefficients object
		 * 
		 */
		virtual void setCellPermeableBiasCoefficients();
		/**
		 * @brief Set the Cell Permeable Coefficients object
		 * 
		 */
		virtual void setCellPermeableCoefficients();
		/**
		 * @brief Get the Cell Outward Flux object
		 * 
		 * @param _cell 
		 * @param _fieldIndex 
		 * @return double 
		 */
		double getCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex);
		/**
		 * @brief Get the Cell Outward Flux object
		 * 
		 * @param _cell 
		 * @param _fieldName 
		 * @return double 
		 */
		virtual double getCellOutwardFlux(const CellG *_cell, std::string _fieldName) { return getCellOutwardFlux(_cell, getFieldIndexByName(_fieldName)); }
		void setCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex, float _outwardFlux);
		virtual void setCellOutwardFlux(const CellG * _cell, std::string _fieldName, float _outwardFlux) { setCellOutwardFlux(_cell, getFieldIndexByName(_fieldName), _outwardFlux); }


		// Solver functions
		bool isValid(Point3D pt) { try { pt2ind(pt); return true; } catch (BasicException) { return false; } }
		virtual Point3D getLatticePointFromPhys(std::vector<float> _physPt) { return Point3D((int)(_physPt[0] / lengthX), (int)(_physPt[1] / lengthY), (int)(_physPt[2] / lengthZ)); }
		unsigned int getSurfaceIndexByName(std::string _surfaceName);
		unsigned int getFieldIndexByName(std::string _fieldName);
		void setUnitsTime(std::string _unitsTime = "s");
		std::string getUnitsTime() { return unitsTime; }
		unsigned int getIndexSurfToCoord(unsigned int _surfaceIndex) { return indexMapSurfToCoord[_surfaceIndex]; }
		int getSurfaceNormSign(unsigned int _surfaceIndex) { return surfaceNormSign[_surfaceIndex]; }
		virtual float getLength(int _dimIndex) { return std::vector<float>{ lengthX, lengthY, lengthZ }[_dimIndex]; }
		virtual float getLengthBySurfaceIndex(unsigned int _surfaceIndex) { return getLength(indexMapSurfToCoord[_surfaceIndex]); }
		float getSignedDistanceBySurfaceIndex(unsigned int _surfaceIndex) { return signedDistanceBySurfaceIndex[_surfaceIndex]; }
		std::vector<float> getLengths() { return std::vector<float>{lengthX, lengthY, lengthZ}; }
		virtual void setLengths(float _length) { setLengths(_length, _length, _length); }
		virtual void setLengths(float _lengthX, float _lengthY, float _lengthZ);
		virtual float getSurfaceArea(unsigned int _surfaceIndex) { return surfaceAreas[_surfaceIndex]; }
		virtual void updateSurfaceAreas();
		double getTimeStep() { return incTime; }
		double getIntegrationTimeStep() { return integrationTimeStep; }
		double getPhysicalTime() { return physTime; }
		virtual ReactionDiffusionSolverFV * getFieldFV(const Point3D &_pt) { return getFieldFV(pt2ind(_pt)); }
		virtual ReactionDiffusionSolverFV * getFieldFV(unsigned int _ind) { return fieldFVs[_ind]; }
		virtual ReactionDiffusionSolverFV * getFieldFV(std::vector<float> _physPt) { return getFieldFV(getLatticePointFromPhys(_physPt)); }
		void setFieldFV(Point3D _pt, ReactionDiffusionSolverFV *_fv) { setFieldFV(pt2ind(_pt), _fv); }
		void setFieldFV(unsigned int _ind, ReactionDiffusionSolverFV *_fv) { fieldFVs[_ind] = _fv; }
		std::map<unsigned int, ReactionDiffusionSolverFV *> getFVNeighborFVs(ReactionDiffusionSolverFV *_fv);
		float getMaxStableTimeStep();
		Point3D ind2pt(unsigned int _ind);
		unsigned int pt2ind(const Point3D &_pt, Dim3D _fieldDim);
		unsigned int pt2ind(const Point3D &_pt) { return pt2ind(_pt, fieldDim); }
		/**
		 * @brief Get the Field Dim object
		 * 
		 * @return Dim3D 
		 */
		Dim3D getFieldDim() { return fieldDim; }

		// Solver interface
		void setFieldExpressionMultiplier(unsigned int _fieldIndex, std::string _expr);
		virtual void setFieldExpressionMultiplier(std::string _fieldName, std::string _expr) { setFieldExpressionMultiplier(getFieldIndexByName(_fieldName), _expr); }
		void setFieldExpressionIndependent(unsigned int _fieldIndex, std::string _expr);
		virtual void setFieldExpressionIndependent(std::string _fieldName, std::string _expr) { setFieldExpressionIndependent(getFieldIndexByName(_fieldName), _expr); }
		std::string getFieldSymbol(unsigned int _fieldIndex) { return fieldSymbolsVec[_fieldIndex]; }
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
		std::map<unsigned int, ReactionDiffusionSolverFV *> neighborFVs;

		double physTime;
		bool stabilityMode;

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
		double getECMaterialDiffusivity(unsigned int _fieldIndex);
		double getFieldDiffusivityField(unsigned int _fieldIndex);
		double getFieldDiffusivityInMedium(unsigned int _fieldIndex);

		
		std::vector<double> diffusiveSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> permeableSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedSurfaceFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);
		std::vector<double> fixedFVConcentrationFlux(unsigned int _fieldIndex, unsigned int _surfaceIndex, ReactionDiffusionSolverFV *_nFv);

	public:

		ReactionDiffusionSolverFV() {};
		ReactionDiffusionSolverFV(ReactionDiffusionSolverFVM *_solver, Point3D _coords, int _numFields);
		virtual ~ReactionDiffusionSolverFV() {};

		const Point3D getCoords() { return coords; }

		void initialize();
		void solve();
		double solveStable();
		void update(double _incTime);
		double getMaxStableTimeStep();

		void clearNeighbors() { neighborFVs.clear(); }
		void addNeighbor(ReactionDiffusionSolverFV *_fv, unsigned int _surfaceIndex) { neighborFVs.insert(make_pair(_surfaceIndex, _fv)); }
		std::map<unsigned int, ReactionDiffusionSolverFV *> getNeighbors() { return neighborFVs; }

		void useConstantDiffusivity(unsigned int _fieldIndex);
		void useConstantDiffusivity(unsigned int _fieldIndex, double _constantDiff);
		void useConstantDiffusivityById(unsigned int _fieldIndex);
		void useConstantDiffusivityById(unsigned int _fieldIndex, double _constantDiff);
		void useECMaterialDiffusivity(unsigned int _fieldIndex) { fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getECMaterialDiffusivity; }
		void useFieldDiffusivityEverywhere(unsigned int _fieldIndex) { fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getFieldDiffusivityField; }
		void useFieldDiffusivityInMedium(unsigned int _fieldIndex) { fieldDiffusivityFunctionPtrs[_fieldIndex] = &ReactionDiffusionSolverFV::getFieldDiffusivityInMedium; }
		
		void useDiffusiveSurfaces(unsigned int _fieldIndex);
		void usePermeableSurfaces(unsigned int _fieldIndex, bool _activate = true);
		void useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _fluxVal);
		void useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, double _val);
		void useFixedFVConcentration(unsigned int _fieldIndex, double _val);

		void registerFieldSymbol(unsigned int _fieldIndex, std::string _fieldSymbol);
		void setDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr);
		void setOffDiagonalFunctionExpression(unsigned int _fieldIndex, std::string _expr);

		mu::Parser zeroMuParserFunction();
		mu::Parser templateParserFunction();
		
		void setConcentrationVec(std::vector<double> _concentrationVec);
		void setConcentration(unsigned int _fieldIndex, double _concentrationVal) { concentrationVecOld[_fieldIndex] = _concentrationVal; }
		void addConcentrationVecIncrement(std::vector<double> _concentrationVecInc);
		std::vector<double> getConcentrationVec() { return concentrationVecOld; }
		double getConcentration(unsigned int _fieldIndex) { return concentrationVecOld[_fieldIndex]; }
		double getConcentrationOld(unsigned int _fieldIndex) { return concentrationVecOld[_fieldIndex]; }

		double getFieldDiffusivity(unsigned int _fieldIndex);

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

		std::vector<double> massConsCorrectionFactors;
		std::vector<double> concentrationVecCopies;

	};

};
#endif