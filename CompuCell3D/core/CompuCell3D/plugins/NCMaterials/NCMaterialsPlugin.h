/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/

#ifndef NCMATERIALSPLUGIN_H
#define NCMATERIALSPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include "NCMaterialsData.h"
#include "NCMaterialsDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {

	/**
	@author T.J. Sego, Ph.D.
	*/

    class Simulator;

	class Potts3D;
	class Automaton;
	class BoundaryStrategy;
    class ParallelUtilsOpenMP;
	class NCMaterialsSteppable;

	class CellG;

	template <class T> class WatchableField3D;

	class DiffusionSolverFE_CPU;
	class ReactionDiffusionSolverFE;

	// For passing intracellular responses due to field interactions
	class NCMATERIALS_EXPORT NCMaterialsCellResponse {
	public:

		NCMaterialsCellResponse(CellG *_cell, std::string _action, std::string _cellTypeDiff = "") :
			cell(_cell),
			Action(_action),
			CellTypeDiff(_cellTypeDiff)
		{}
		~NCMaterialsCellResponse() {};

		CellG *cell;
		std::string Action;
		std::string CellTypeDiff;
	};

	class NCMATERIALS_EXPORT NCMaterialsPlugin : public Plugin, public EnergyFunction, public CellGChangeWatcher {
	private:
		BasicClassAccessor<NCMaterialsData> NCMaterialsDataAccessor;
		BasicClassAccessor<NCMaterialComponentData> NCMaterialComponentDataAccessor;
		BasicClassAccessor<NCMaterialCellData> NCMaterialCellDataAccessor;

		CC3DXMLElement *xmlData;
		Potts3D *potts;
		Simulator *sim;
	    ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

		unsigned int numberOfMaterials;
		std::vector<std::string> fieldNameVec;

		double depth;

		BasicClassAccessor<NCMaterialsData> * NCMaterialsDataAccessorPtr;
		BasicClassAccessor<NCMaterialComponentData> * NCMaterialComponentDataAccessorPtr;
		BasicClassAccessor<NCMaterialCellData> * NCMaterialCellDataAccessorPtr;

		Automaton *automaton;
		bool weightDistance;

		DiffusionSolverFE_CPU *pdeSolverFE_CPU;
		ReactionDiffusionSolverFE *pdeSolverRDFE;

		unsigned int maxNeighborIndex; // for first order neighborhood, used in material advection
		unsigned int maxNeighborIndexAdh; // for adhesion neighborhoods
		BoundaryStrategy *boundaryStrategy;
		BoundaryStrategy *boundaryStrategyAdh;

		bool NCMaterialsInitialized;
		std::vector<NCMaterialComponentData> NCMaterialsVec;
		std::map<std::string, int> NCMaterialNameIndexMap;
		std::map<std::string, std::vector<float> > typeToRemodelingQuantityMap;
		std::vector<std::vector<float> > AdhesionCoefficientsByTypeId;

		CC3DXMLElementList NCMaterialAdhesionXMLVec;
		CC3DXMLElementList NCMaterialRemodelingQuantityXMLVec;
		std::vector<std::string> cellTypeNamesByTypeId;
		std::vector<std::vector<float> > RemodelingQuantityByTypeId;

		std::map<std::string, bool> variableDiffusivityFieldFlagMap; // true when variable diffusion coefficient is defined for a field
		std::map<std::string, float> scalingExtraMCSVec;

		Dim3D fieldDim;
		WatchableField3D<NCMaterialsData *> *NCMaterialsField; // maybe add watcher interface in future dev.

		NCMaterialsSteppable *ncMaterialsSteppable; // partner steppable

		std::vector<NCMaterialsCellResponse> cellResponses; // populated by steppable and passed to Python

		int dotProduct(Point3D _pt1, Point3D _pt2);

	public:

		NCMaterialsPlugin();
		virtual ~NCMaterialsPlugin();

		BasicClassAccessor<NCMaterialsData> * getNCMaterialsDataAccessorPtr() { return &NCMaterialsDataAccessor; }
		BasicClassAccessor<NCMaterialComponentData> * getNCMaterialComponentDataAccessorPtr() { return &NCMaterialComponentDataAccessor; }
		BasicClassAccessor<NCMaterialCellData> * getNCMaterialCellDataAccessorPtr() { return &NCMaterialCellDataAccessor; }

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

		virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent & _event);

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();

		// cell-NCmaterial contact effective energy term
		double NCMaterialContactEnergy(const CellG *cell, std::vector<float> _qtyVec);
		double NCMaterialContactEnergyChange(const CellG *cellNew, std::vector<float> _qtyVecNew, const CellG *cellOld, std::vector<float> _qtyVecOld);
		// NCmaterial durability effective energy term
		double NCMaterialDurabilityEnergy(std::vector<float> _qtyVec);

		void initializeNCMaterialsField(Dim3D _fieldDim, bool _resetting = false);
		void initializeNCMaterialsField(bool _resetting = false) { initializeNCMaterialsField(fieldDim, _resetting); }
		void deleteNCMaterialsField();

		void initializeNCMaterials();
		std::vector<NCMaterialComponentData> getNCMaterialsVec() { return NCMaterialsVec; }
		std::vector<NCMaterialComponentData> *getNCMaterialsVecPtr() { return &NCMaterialsVec; }
		unsigned int getNumberOfMaterials() { return numberOfMaterials; }
		std::map<std::string, int> getNCMaterialNameIndexMap() { return NCMaterialNameIndexMap; }
		void setMaterialNameVector();
		std::vector<std::string> getMaterialNameVector() { 
			setMaterialNameVector();
			return fieldNameVec;
		}
		std::vector<float> checkQuantities(std::vector<float> _qtyVec);

		// functions used to manipulate extracellular material cell definitions

		void setRemodelingQuantityByName(const CellG * _cell, std::string _NCMaterialName, float _quantity);
		void setRemodelingQuantityByIndex(const CellG * _cell, int _idx, float _quantity);
		void setRemodelingQuantityVector(const CellG * _cell, std::vector<float> _quantityVec);
		void assignNewRemodelingQuantityVector(const CellG * _cell, int _numMtls=-1);
		//		medium functions

		void setMediumNCMaterialQuantityByName(const Point3D &pt, std::string _NCMaterialName, float _quantity);
		void setMediumNCMaterialQuantityByIndex(const Point3D &pt, int _idx, float _quantity);
		void setMediumNCMaterialQuantityVector(const Point3D &pt, std::vector<float> _quantityVec);
		void assignNewMediumNCMaterialQuantityVector(const Point3D &pt, int _numMtls);
		//		material functions

		void setNCMaterialDurabilityByName(std::string _NCMaterialName, float _durabilityLM);
		void setNCMaterialDurabilityByIndex(int _idx, float _durabilityLM);
		void setNCMaterialAdvectingByName(std::string _NCMaterialName, bool _isAdvecting);
		void setNCMaterialAdvectingByIndex(int _idx, bool _isAdvecting);
		//		adhesion functions

		void setNCAdhesionByCell(const CellG *_cell, std::vector<float> _adhVec);
		void setNCAdhesionByCellAndMaterialIndex(const CellG *_cell, int _idx, float _val);
		void setNCAdhesionByCellAndMaterialName(const CellG *_cell, std::string _NCMaterialName, float _val);

        // functions used to retrieve extracellular material cell definitions

		float getRemodelingQuantityByName(const CellG * _cell, std::string _NCMaterialName);
		float getRemodelingQuantityByIndex(const CellG * _cell, int _idx);
		std::vector<float> getRemodelingQuantityVector(const CellG * _cell);
		//		medium functions

		float getMediumNCMaterialQuantityByName(const Point3D &pt, std::string _NCMaterialName);
		float getMediumNCMaterialQuantityByIndex(const Point3D &pt, int _idx);
		std::vector<float> getMediumNCMaterialQuantityVector(const Point3D &pt);
		std::vector<float> getMediumAdvectingNCMaterialQuantityVector(const Point3D &pt);
		//		material functions

		float getNCMaterialDurabilityByName(std::string _NCMaterialName);
		float getNCMaterialDurabilityByIndex(int _idx);
		bool getNCMaterialAdvectingByName(std::string _NCMaterialName);
		bool getNCMaterialAdvectingByIndex(int _idx);
		virtual int getNCMaterialIndexByName(std::string _NCMaterialName);
		virtual std::string getNCMaterialNameByIndex(int _idx) {
			for (std::map<std::string, int>::iterator mitr = NCMaterialNameIndexMap.begin(); mitr != NCMaterialNameIndexMap.end(); ++mitr) if (mitr->second == _idx) return mitr->first;
			return "";
		}
		//		adhesion functions

		std::vector<float> getNCAdhesionByCell(const CellG *_cell);
		std::vector<float> getNCAdhesionByCellTypeId(int _idx);
		float getNCAdhesionByCellAndMaterialIndex(const CellG *_cell, int _idx);
		float getNCAdhesionByCellAndMaterialName(const CellG *_cell, std::string _NCMaterialName);

		// calculates material quantity vector when source is a medium site
		std::vector<float> calculateCopyQuantityVec(const CellG * _cell, const Point3D &pt);

		// Returns flag for variable field diffusion coefficient
		// Fields that aren't registered return false
		bool getVariableDiffusivityFieldFlag(std::string _fieldName) { 
			std::map<std::string, bool>::iterator mitr = variableDiffusivityFieldFlagMap.find(_fieldName);
			if (mitr == variableDiffusivityFieldFlagMap.end()) return false;
			else return variableDiffusivityFieldFlagMap[_fieldName]; 
		};
		// Sets flag for variable field diffusion coefficient
		void setVariableDiffusivityFieldFlagMap(std::string _fieldName, bool _flag=true) {
			std::map<std::string, bool>::iterator mitr = variableDiffusivityFieldFlagMap.find(_fieldName);
			if (mitr == variableDiffusivityFieldFlagMap.end()) variableDiffusivityFieldFlagMap.insert(make_pair(_fieldName, _flag));
			else variableDiffusivityFieldFlagMap[_fieldName] = _flag;
		}
		// Returns diffusion coefficient at point pt for field _fieldName
		// If site is intracellular, returns 0.0
		float getLocalDiffusivity(const Point3D &pt, std::string _fieldName);
		// Returns maxmium diffusion coefficient for field _fieldName
		float getMaxFieldDiffusivity(std::string _fieldName);
		void setScalingExtraMCSVec(std::string _fieldName, float _scalingFactor) { scalingExtraMCSVec[_fieldName] = _scalingFactor; }

		// material field drawing functions

		// Draw a parallelepiped of _qtyVec from _startPos defined by vectors _lenVec1, _lenVec2, and _lenVec3
		// _lenVec components equal to -1 go to the end of the domain
		void ParaDraw(std::vector<float> _qtyVec, Point3D _startPos = { 0,0,0 }, Point3D _lenVec1 = { -1,-1,-1 }, Point3D _lenVec2 = { -1,-1,-1 }, Point3D _lenVec3 = { -1,-1,-1 });
		// Draw a cylinder of _qtyVec from _startPos defined by radius _radius and vector _lenVec
		// _lenVec components equal to -1 go to the end of the domain
		void CylinderDraw(std::vector<float> _qtyVec, short _radius, Point3D _startPos = { 0,0,0 }, Point3D _lenVec = { -1,-1,-1 });
		// Draw an ellipsoid of _qtyVec about _center with axis lengths _lenVec rotated by angles _angleVec
		// Rotations occur with the order _angleVec[2], _angleVec[1], _angleVec[0]
		void EllipsoidDraw(std::vector<float> _qtyVec, Point3D _center = { 0,0,0 }, std::vector<short> _lenVec = std::vector<short>(3, 1.0), std::vector<double> _angleVec = std::vector<double>(3, 0.0));
		// Draw an ellipse of _qtyVec about _center rotated by angle _angle (in degrees) with semimajor axis length _length and eccentricity _eccentricity
		void EllipseDraw(std::vector<float> _qtyVec, int _length, Point3D _center = { 0,0,0 }, double _angle = 0.0, double _eccentricity = 0.0);

		// Returns pointer to material field
		virtual Field3D<NCMaterialsData *> *getNCMaterialField() { return (Field3D<NCMaterialsData *> *)NCMaterialsField; }

		// Sets partner steppable
		void setNCMaterialsSteppable(NCMaterialsSteppable *_ncMaterialsSteppable) { ncMaterialsSteppable = _ncMaterialsSteppable; }

		// Interface with python

		int numberOfResponsesOccurred() { return (int)cellResponses.size(); };
		CellG *getCellResponseCell(int _idx) { return cellResponses[_idx].cell; };
		std::string getCellResponseAction(int _idx) { return cellResponses[_idx].Action; };
		std::string getCellResponseCellTypeDiff(int _idx) { return cellResponses[_idx].CellTypeDiff; };

		// Interface with steppable

		virtual void resetCellResponse() { cellResponses.clear(); }
		virtual void addCellResponse(CellG *_cell, std::string _action, std::string _cellTypeDiff = "") { cellResponses.push_back(NCMaterialsCellResponse(_cell, _action, _cellTypeDiff)); };

		// Interface between steppable and python

		float calculateTotalInterfaceQuantityByMaterialIndex(CellG *cell, int _materialIdx);
		float calculateTotalInterfaceQuantityByMaterialName(CellG *cell, std::string _materialName);
		float calculateCellProbabilityProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		float calculateCellProbabilityDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		float calculateCellProbabilityDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		float calculateCellProbabilityAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		bool getCellResponseProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		bool getCellResponseDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		bool getCellResponseDifferentiation(CellG *cell, std::string newCellType = std::string(), Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		bool getCellResponseAsymmetricDivision(CellG *cell, std::string newCellType = std::string(), Field3D<NCMaterialsData *> *_ncmaterialsField = 0);

		void overrideInitialization();

	};
}
#endif
