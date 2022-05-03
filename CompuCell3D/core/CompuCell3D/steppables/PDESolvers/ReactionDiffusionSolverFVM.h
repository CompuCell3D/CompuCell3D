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
		 * @brief 
		 * 
		 */
		void loadCellData();
		void loadFieldExpressions();
		void loadFieldExpressionMultiplier(unsigned int _fieldIndex);
		void loadFieldExpressionIndependent(unsigned int _fieldIndex);
		virtual void loadFieldExpressionMultiplier(std::string _fieldName) { loadFieldExpressionMultiplier(getFieldIndexByName(_fieldName)); }
		virtual void loadFieldExpressionIndependent(std::string _fieldName) { loadFieldExpressionIndependent(getFieldIndexByName(_fieldName)); }
		virtual void loadFieldExpressionMultiplier(std::string _fieldName, std::string _expr);
		virtual void loadFieldExpressionIndependent(std::string _fieldName, std::string _expr);
		void loadFieldExpressionMultiplier(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv);
		void loadFieldExpressionIndependent(unsigned int _fieldIndex, ReactionDiffusionSolverFV *_fv);
		void initializeFVs(Dim3D _fieldDim);
		void initializeFieldUsingEquation(unsigned int _fieldIndex, std::string _expr);
		virtual void initializeFieldUsingEquation(std::string _fieldName, std::string _expr) { initializeFieldUsingEquation(getFieldIndexByName(_fieldName), _expr); }
		std::vector<double> totalCellConcentration(const CellG *_cell);
		std::vector<double> totalMediumConcentration();
		void updateTotalCellConcentrations();
		void updateTotalConcentrations();
		std::vector<Point3D> getCellPixelVec(const CellG *_cell);
		std::vector<Point3D> getMediumPixelVec();
		std::vector<double> totalPixelSetConcentration(std::vector<Point3D> _pixelVec);
		
		// FV interface
		Point3D getCoordsOfFV(ReactionDiffusionSolverFV *_fv);
		CellG * FVtoCellMap(ReactionDiffusionSolverFV * _fv);

		double getConstantFieldDiffusivity(unsigned int _fieldIndex) { return constantDiffusionCoefficientsVec[_fieldIndex]; }
		// float getECMaterialDiffusivity(unsigned int _fieldIndex, Point3D coords) { return ecMaterialsPlugin->getLocalDiffusivity(coords, concentrationFieldNameVector[_fieldIndex]); }
		float getECMaterialDiffusivity(unsigned int _fieldIndex, Point3D coords) { return 0.0; } // To be implemented with commented declaration/definition
		float getDiffusivityFieldPtVal(unsigned int _fieldIndex, const Point3D &pt) { return diffusivityFieldIndexToFieldMap[_fieldIndex]->get(pt); }

		void useConstantDiffusivity(unsigned int _fieldIndex) { useConstantDiffusivity(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }
		void useConstantDiffusivity(unsigned int _fieldIndex, double _diffusivityCoefficient);
		virtual void useConstantDiffusivity(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivity(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		void useConstantDiffusivityByType(unsigned int _fieldIndex) { useConstantDiffusivityByType(_fieldIndex, constantDiffusionCoefficientsVec[_fieldIndex]); }
		void useConstantDiffusivityByType(unsigned int _fieldIndex, double _diffusivityCoefficient);
		virtual void useConstantDiffusivityByType(std::string _fieldName, double _diffusivityCoefficient) { useConstantDiffusivityByType(getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		void useFieldDiffusivityInMedium(unsigned int _fieldIndex);
		virtual void useFieldDiffusivityInMedium(std::string _fieldName) { useFieldDiffusivityInMedium(getFieldIndexByName(_fieldName)); }
		void useFieldDiffusivityEverywhere(unsigned int _fieldIndex);
		virtual void useFieldDiffusivityEverywhere(std::string _fieldName) { useFieldDiffusivityEverywhere(getFieldIndexByName(_fieldName)); }
		void initDiffusivityField(unsigned int _fieldIndex);
		virtual void initDiffusivityField(std::string _fieldName) { initDiffusivityField(getFieldIndexByName(_fieldName)); }
		void setDiffusivityFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);
		virtual void setDiffusivityFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setDiffusivityFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }

		float getConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt);
		float getConcentrationFieldPtVal(std::string _fieldName, Point3D _pt) { return getConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt); }
		void setConcentrationFieldPtVal(unsigned int _fieldIndex, Point3D _pt, float _val);
		void setConcentrationFieldPtVal(std::string _fieldName, Point3D _pt, float _val) { setConcentrationFieldPtVal(getFieldIndexByName(_fieldName), _pt, _val); }

		void useFixedFluxSurface(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _outwardFluxVal, ReactionDiffusionSolverFV *_fv);
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, Point3D _pt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_pt)); }
		virtual void useFixedFluxSurface(std::string _fieldName, std::string _surfaceName, float _outwardFluxVal, std::vector<float> _physPt) { useFixedFluxSurface(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _outwardFluxVal, getFieldFV(_physPt)); }
		void useFixedConcentration(unsigned int _fieldIndex, unsigned int _surfaceIndex, float _val, ReactionDiffusionSolverFV *_fv);
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, Point3D _pt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_pt)); }
		virtual void useFixedConcentration(std::string _fieldName, std::string _surfaceName, float _val, std::vector<float> _physPt) { useFixedConcentration(getFieldIndexByName(_fieldName), getSurfaceIndexByName(_surfaceName), _val, getFieldFV(_physPt)); }
		void useFixedFVConcentration(unsigned int _fieldIndex, float _val, ReactionDiffusionSolverFV *_fv);
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, Point3D _pt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_pt)); }
		virtual void useFixedFVConcentration(std::string _fieldName, float _val, std::vector<float> _physPt) { useFixedFVConcentration(getFieldIndexByName(_fieldName), _val, getFieldFV(_physPt)); }

		void useDiffusiveSurfaces(unsigned int _fieldIndex);
		virtual void useDiffusiveSurfaces(std::string _fieldName) { useDiffusiveSurfaces(getFieldIndexByName(_fieldName)); }
		void usePermeableSurfaces(unsigned int _fieldIndex) { usePermeableSurfaces(_fieldIndex, true); }
		void usePermeableSurfaces(unsigned int _fieldIndex, bool _activate);
		virtual void usePermeableSurfaces(std::string _fieldName, bool _activate) { usePermeableSurfaces(getFieldIndexByName(_fieldName), _activate); }

		// Cell interface
		void initializeCellData(const CellG *_cell, unsigned int _numFields);
		void initializeCellData(unsigned int _numFields);

		double getCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		virtual double getCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { return getCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex, double _diffusivityCoefficient);
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName, double _diffusivityCoefficient) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName), _diffusivityCoefficient); }
		void setCellDiffusivityCoefficient(const CellG * _cell, unsigned int _fieldIndex);
		virtual void setCellDiffusivityCoefficient(const CellG * _cell, std::string _fieldName) { setCellDiffusivityCoefficient(_cell, getFieldIndexByName(_fieldName)); }
		void setCellDiffusivityCoefficients(unsigned int _fieldIndex);
		virtual void setCellDiffusivityCoefficients(std::string _fieldName) { setCellDiffusivityCoefficients(getFieldIndexByName(_fieldName)); }
		virtual void setCellDiffusivityCoefficients();

		std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, unsigned int _fieldIndex);
		virtual std::vector<double> getPermeableCoefficients(const CellG * _cell, const CellG * _nCell, std::string _fieldName) {
			return getPermeableCoefficients(_cell, _nCell, getFieldIndexByName(_fieldName)); }
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeationCoefficient);
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeationCoefficient) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeationCoefficient); }
		void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		virtual void setCellPermeationCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeationCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		void setCellPermeationCoefficients(unsigned int _fieldIndex);
		virtual void setCellPermeationCoefficients(std::string _fieldName) { setCellPermeationCoefficients(getFieldIndexByName(_fieldName)); }
		virtual void setCellPermeationCoefficients();

		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex, double _permeableBiasCoefficient);
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName, double _permeableBiasCoefficient) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName), _permeableBiasCoefficient); }
		void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, unsigned int _fieldIndex);
		virtual void setCellPermeableBiasCoefficient(const CellG * _cell, unsigned int _nCellTypeId, std::string _fieldName) { setCellPermeableBiasCoefficient(_cell, _nCellTypeId, getFieldIndexByName(_fieldName)); }
		void setCellPermeableBiasCoefficients(unsigned int _fieldIndex);
		virtual void setCellPermeableBiasCoefficients(std::string _fieldName) { setCellPermeableBiasCoefficients(getFieldIndexByName(_fieldName)); }
		virtual void setCellPermeableBiasCoefficients();

		virtual void setCellPermeableCoefficients();

		double getCellOutwardFlux(const CellG *_cell, unsigned int _fieldIndex);
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