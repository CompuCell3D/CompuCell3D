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

#ifndef NCMATERIALSSTEPPABLE_H
#define NCMATERIALSSTEPPABLE_H

#include <ppl.h>
#include <concurrent_vector.h>

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include "NCMaterialsSteppableDLLSpecifier.h"
#include "CompuCell3D/plugins/NCMaterials/NCMaterialsPlugin.h"
#include "CompuCell3D/plugins/NCMaterials/NCMaterialsData.h"
#include "CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h"
#include "CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h"

namespace CompuCell3D {

	/**
	@author T.J. Sego, Ph.D.
	*/

    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
	class ParallelUtilsOpenMP;
    class CellInventory;
    class CellG;
	class BoundaryPixelTrackerData;
	class BoundaryPixelTrackerPlugin;
	class NCMaterialsPlugin;
	class NCMaterialsData;

	class NCMATERIALSSTEPPABLE_EXPORT NCMaterialsSteppable : public Steppable {
        WatchableField3D<CellG *> *cellFieldG;
        Simulator *sim;
        Potts3D *potts;
        CC3DXMLElement *xmlData;
        Automaton *automaton;
        BoundaryStrategy *boundaryStrategy;
        CellInventory *cellInventoryPtr;
		ParallelUtilsOpenMP *pUtils;
		ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

		BoundaryPixelTrackerPlugin *boundaryTrackerPlugin;
		NCMaterialsPlugin *ncMaterialsPlugin;

		int neighborOrder = 1;
		int nNeighbors;
		Dim3D fieldDim;

		bool NCMaterialsInitialized;
		int numberOfMaterials;
		Field3D<NCMaterialsData *> *NCMaterialsField;
		std::vector<NCMaterialComponentData> *NCMaterialsVec;

		int numberOfFields;
		std::vector<Field3D<float> *> fieldVec;
		std::map<std::string, int> fieldNames;

		int numberOfCellTypes;
		std::map<std::string, int> cellTypeNames;

		bool MaterialInteractionsDefined;
		bool MaterialReactionsDefined;
		bool MaterialDiffusionDefined;
		bool FieldInteractionsDefined;
		bool CellInteractionsDefined;
		bool AnyInteractionsDefined;

		std::vector<int> idxMaterialReactionsDefined;
		std::vector<std::vector<int> > idxidxMaterialReactionsDefined;
		std::vector<int> idxMaterialDiffusionDefined;
		std::vector<int> idxToFieldInteractionsDefined;
		std::vector<std::vector<int> > idxidxToFieldInteractionsDefined;
		std::vector<int> idxFromFieldInteractionsDefined;
		std::vector<std::vector<int> > idxidxFromFieldInteractionsDefined;
		std::vector<int> idxCellInteractionsDefined;

		std::vector<std::vector<float> > toFieldReactionCoefficientsByIndex;
		std::vector<std::vector<float> > fromFieldReactionCoefficientsByIndex;
		std::vector<std::vector<std::vector<float> > > materialReactionCoefficientsByIndex;
		std::vector<std::vector<float> > CellTypeCoefficientsProliferationByIndex;
		std::vector<std::vector<std::vector<float> > > CellTypeCoefficientsProliferationAsymByIndex;
		std::vector<std::vector<std::vector<float> > > CellTypeCoefficientsDifferentiationByIndex;
		std::vector<std::vector<float> > CellTypeCoefficientsDeathByIndex;

		// Utility functions
		float randFloatGen01() { return float(rand()) / (float(RAND_MAX) + 1); };
		std::vector<std::vector<int> > permutations(int _numberOfVals);
		bool testProb(float prob) { return prob > FLT_MIN && prob > randFloatGen01(); };

	public:

        NCMaterialsSteppable();
        virtual ~NCMaterialsSteppable();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator) {};
		void handleEvent(CC3DEvent & _event);

        // Steppable interface
		virtual void start() {};
        virtual void step(const unsigned int currentStep);
        virtual void finish() {}

        // SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
        virtual std::string steerableName();
        virtual std::string toString();

		// Plugin wrappers
		int getNCMaterialIndexByName(std::string _NCMaterialName) { return ncMaterialsPlugin->getNCMaterialIndexByName(_NCMaterialName); };
		std::string getNCMaterialNameByIndex(int _idx) { return ncMaterialsPlugin->getNCMaterialNameByIndex(_idx); };
		std::vector<float> checkQuantities(std::vector<float> _qtyVec) { return ncMaterialsPlugin->checkQuantities(_qtyVec); };
		void setMediumNCMaterialQuantityVector(const Point3D &pt, std::vector<float> _quantityVec) { ncMaterialsPlugin->setMediumNCMaterialQuantityVector(pt, checkQuantities(_quantityVec)); };

		// Steppable methods
		void constructFieldReactionCoefficients();
		void constructMaterialReactionCoefficients();
		void constructCellTypeCoefficients();
		void constructCellTypeCoefficientsProliferation();
		void constructCellTypeCoefficientsDifferentiation();
		void constructCellTypeCoefficientsDeath();

		void calculateMaterialToFieldInteractions(const Point3D &pt, std::vector<float> _qtyOld);
		void calculateCellInteractions(Field3D<NCMaterialsData *> *NCMaterialsField);
		std::tuple<float, float, std::vector<float>, std::vector<float> > calculateCellProbabilities(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0, std::vector<bool> _resultsSelect = std::vector<bool>(4, true));

		int getCellTypeIndexByName(std::string _cellTypeName) {
			std::map<std::string, int>::iterator mitr = cellTypeNames.find(_cellTypeName);
			if (mitr == cellTypeNames.end()) return -1;
			else return mitr->second;
		};
		int getFieldIndexByName(std::string _fieldName) {
			std::map<std::string, int>::iterator mitr = fieldNames.find(_fieldName);
			if (mitr == fieldNames.end()) return -1;
			else return mitr->second;
		}
		std::string getFieldNameByIndex(int _idx) {
			for (std::map<std::string, int>::iterator mitr = fieldNames.begin(); mitr != fieldNames.end(); ++mitr) if (mitr->second == _idx) return mitr->first;
			return "";
		}

		// Interface with plugin
		virtual float calculateTotalInterfaceQuantityByMaterialIndex(CellG *cell, int _materialIdx, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual float calculateTotalInterfaceQuantityByMaterialName(CellG *cell, std::string _materialName, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual float calculateCellProbabilityProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual float calculateCellProbabilityDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual float calculateCellProbabilityDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual float calculateCellProbabilityAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual bool getCellResponseProliferation(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual bool getCellResponseDeath(CellG *cell, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual bool getCellResponseDifferentiation(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);
		virtual bool getCellResponseAsymmetricDivision(CellG *cell, std::string newCellType, Field3D<NCMaterialsData *> *_ncmaterialsField = 0);


		Point3D ind2pt(unsigned int _ind);
		unsigned int pt2ind(const Point3D &_pt, Dim3D _fieldDim);
		unsigned int pt2ind(const Point3D &_pt) { return pt2ind(_pt, fieldDim); }
    };

};

#endif