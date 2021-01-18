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

#ifndef FOCALPOINTPLACTICITYPLUGIN_H
#define FOCALPOINTPLACTICITYPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "FocalPointPlasticityLinkInventory.h"
#include "FocalPointPlasticityDLLSpecifier.h"

class CC3DXMLElement;

//Note: Target distance is set to be 0.9 * current distance between cells. target distance from xml is ignored

namespace CompuCell3D {
    class Simulator;
    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class ParallelUtilsOpenMP;

    class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityPlugin : public Plugin, public EnergyFunction, public CellGChangeWatcher {


        BasicClassAccessor<FPPLinkInventoryTracker<FocalPointPlasticityLink> > cellLinkInventoryTracker;
        BasicClassAccessor<FPPLinkInventoryTracker<FocalPointPlasticityInternalLink> > cellInternalLinkInventoryTracker;
        BasicClassAccessor<FPPLinkInventoryTracker<FocalPointPlasticityAnchor> > cellAnchorInventoryTracker;

        Simulator *sim;

        Potts3D *potts;

        ParallelUtilsOpenMP *pUtils;
        std::string autoName;
        double depth;

        Automaton *automaton;
        bool weightDistance;
        unsigned int maxNeighborIndex;
        unsigned int maxNeighborIndexJunctionMove;
        BoundaryStrategy * boundaryStrategy;
        CC3DXMLElement *xmlData;
		
		FPPLinkInventory linkInv;
		FPPInternalLinkInventory linkInvInternal;
		FPPAnchorInventory linkInvAnchor;

		std::set<std::string> plasticityTypesNames;
		std::set<unsigned char> plasticityTypes;
		std::set<unsigned char> internalPlasticityTypes;

		Dim3D fieldDim;
		double lambda;

		double activationEnergy;
		double targetDistance;
		double maxDistance;
		double potentialFunction(double _lambda, double _offset, double _targetDistance, double _distance);

		//vectorized variables for convenient parallel access      
		std::vector<short> newJunctionInitiatedFlagVec;
		std::vector<short> newJunctionInitiatedFlagWithinClusterVec;
		std::vector<CellG *> newNeighborVec;

		unsigned int maxNumberOfJunctions;

        enum FunctionType { GLOBAL = 0, BYCELLTYPE = 1, BYCELLID = 2 };

        FunctionType functionType;

        typedef double (FocalPointPlasticityPlugin::*diffEnergyFcnPtr_t)(float _deltaL, float _lBefore, const FocalPointPlasticityTrackerData * _plasticityTrackerData, const CellG *_cell, bool _useCluster);
        diffEnergyFcnPtr_t diffEnergyFcnPtr;


        ExpressionEvaluatorDepot eed;


        typedef double (FocalPointPlasticityPlugin::*constituentLawFcnPtr_t)(float _lambda, float _length, float _targetLength);

        constituentLawFcnPtr_t constituentLawFcnPtr;
        double elasticLinkConstituentLaw(float _lambda, float _length, float _targetLength);
        double customLinkConstituentLaw(float _lambda, float _length, float _targetLength);

        double diffEnergyLocal(float _deltaL, float _lBefore, const FocalPointPlasticityTrackerData * _plasticityTrackerData, const CellG *_cell, bool _useCluster = false);        
        double diffEnergyByType(float _deltaL, float _lBefore, const FocalPointPlasticityTrackerData * _plasticityTrackerData, const CellG *_cell, bool _useCluster = false);

        double tryAddingNewJunction(const Point3D &pt, const CellG *newCell);
        double tryAddingNewJunctionWithinCluster(const Point3D &pt, const CellG *newCell);
  
        typedef std::map<int, FocalPointPlasticityTrackerData> plastParams_t;

        plastParams_t plastParams;
        plastParams_t internalPlastParams;

        plastParams_t typeSpecificPlastParams;
        plastParams_t internalTypeSpecificPlastParams;
      
		typedef std::vector<std::vector<FocalPointPlasticityTrackerData> > FocalPointPlasticityTrackerDataArray_t;
		typedef std::vector<FocalPointPlasticityTrackerData> FocalPointPlasticityTrackerDataVector_t;

        FocalPointPlasticityTrackerDataArray_t plastParamsArray;
        FocalPointPlasticityTrackerDataArray_t internalPlastParamsArray;

        std::vector<int> maxNumberOfJunctionsTotalVec;
        std::vector<int> maxNumberOfJunctionsInternalTotalVec;
        int neighborOrder;

    public:
        FocalPointPlasticityPlugin();
        virtual ~FocalPointPlasticityPlugin();

		//Plugin interface
		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator);
        virtual void handleEvent(CC3DEvent & _event);
		
		//EnergyFunction Interface
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

		// Field3DChangeWatcher interface
		virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

		//used to manually control parameters plasticity term for pair of cells involved
		void setFocalPointPlasticityParameters(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance=0.0,double _maxDistance=0.0);
		void setInternalFocalPointPlasticityParameters(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance=0.0,double _maxDistance=0.0);
		double getPlasticityParametersLambdaDistance(CellG * _cell1,CellG * _cell2);
		double getPlasticityParametersTargetDistance(CellG * _cell1,CellG * _cell2);

		void deleteFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2);
		void deleteInternalFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2);
		void createFocalPointPlasticityLink(CellG * _cell1, CellG * _cell2,double _lambda, double _targetDistance=0.0,double _maxDistance=0.0);
		void createInternalFocalPointPlasticityLink(CellG * _cell1,CellG * _cell2,double _lambda, double _targetDistance=0.0,double _maxDistance=0.0);

		// Inventory accessors

		FPPLinkInventory* getLinkInventory() { return &linkInv; }
		FPPInternalLinkInventory* getInternalLinkInventory() { return &linkInvInternal; }
		FPPAnchorInventory* getAnchorInventory() { return &linkInvAnchor; }
		
		//used for serialization and restart 
		void insertFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd);
		void insertInternalFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd);
		void insertAnchorFPPData(CellG * _cell,FocalPointPlasticityTrackerData * _fpptd);
		std::vector<FocalPointPlasticityTrackerData> getFPPDataVec(CellG * _cell);
		std::vector<FocalPointPlasticityTrackerData> getInternalFPPDataVec(CellG * _cell);
		std::vector<FocalPointPlasticityTrackerData> getAnchorFPPDataVec(CellG * _cell);

		//anchors
		int createAnchor(CellG * _cell, double _lambda, double _targetDistance=0.0,double _maxDistance=100000.0,float _x=0, float _y=0, float _z=0);
		void deleteAnchor(CellG * _cell, int _anchorId);
		void setAnchorParameters(CellG * _cell, int _anchorId,double _lambda, double _targetDistance=0.0,double _maxDistance=100000.0,float _x=-1, float _y=-1, float _z=-1);

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();
	protected:
		int getIndex(const int type1, const int type2) const;

	};
};
#endif
