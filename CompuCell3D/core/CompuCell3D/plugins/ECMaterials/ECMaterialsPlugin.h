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

#ifndef ECMATERIALSPLUGIN_H
#define ECMATERIALSPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include "ECMaterialsData.h"
#include "ECMaterialsDLLSpecifier.h"

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

	class CellG;

	template <class T> class WatchableField3D;
	// template <typename T> class WatchableField3D;

	// class ECMaterialsData;
	// class ECMaterialComponentData;
	// class ECMaterialCellData;

	class ECMATERIALS_EXPORT ECMaterialsPlugin : public Plugin, public EnergyFunction, public CellGChangeWatcher {
	private:
		BasicClassAccessor<ECMaterialsData> ECMaterialsDataAccessor;
		BasicClassAccessor<ECMaterialComponentData> ECMaterialComponentDataAccessor;
		BasicClassAccessor<ECMaterialCellData> ECMaterialCellDataAccessor;
		CC3DXMLElement *xmlData;
		Potts3D *potts;
		Simulator *sim;
	    ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
		//Energy function data

		unsigned int numberOfMaterials;
		std::vector<std::string> fieldNameVec;

		double depth;

		BasicClassAccessor<ECMaterialsData> * ECMaterialsDataAccessorPtr;
		BasicClassAccessor<ECMaterialComponentData> * ECMaterialComponentDataAccessorPtr;
		BasicClassAccessor<ECMaterialCellData> * ECMaterialCellDataAccessorPtr;

		Automaton *automaton;
		bool weightDistance;

		unsigned int maxNeighborIndex; // for first order neighborhood, used in advection
		unsigned int maxNeighborIndexAdh; // for adhesion neighborhoods
		BoundaryStrategy *boundaryStrategy;
		BoundaryStrategy *boundaryStrategyAdh;
		// float energyOffset;

		bool ECMaterialsInitialized;
		std::vector<ECMaterialComponentData> ECMaterialsVec;
		std::map<std::string, int> ECMaterialNameIndexMap;
		std::map<std::string, std::vector<float> > typeToRemodelingQuantityMap;
		std::vector<std::vector<float> > AdhesionCoefficientsByTypeId;

		CC3DXMLElementList ECMaterialAdhesionXMLVec;
		CC3DXMLElementList ECMaterialRemodelingQuantityXMLVec;
		std::vector<std::string> cellTypeNamesByTypeId;
		std::vector<std::vector<float> > RemodelingQuantityByTypeId;

		Dim3D fieldDim;
		WatchableField3D<ECMaterialsData *> *ECMaterialsField;

		int dotProduct(Point3D _pt1, Point3D _pt2);

	public:

		ECMaterialsPlugin();
		virtual ~ECMaterialsPlugin();

		BasicClassAccessor<ECMaterialsData> * getECMaterialsDataAccessorPtr(){return & ECMaterialsDataAccessor;}
		BasicClassAccessor<ECMaterialComponentData> * getECMaterialComponentDataAccessorPtr(){return & ECMaterialComponentDataAccessor;}
		BasicClassAccessor<ECMaterialCellData> * getECMaterialCellDataAccessorPtr(){return & ECMaterialCellDataAccessor;}

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

		virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent & _event);

		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();

		/**
		Return the cell-ECmaterial contact energy term
		*/
		double ECMaterialContactEnergy(const CellG *cell, std::vector<float> _qtyVec);
		/**
		Return the ECmaterial durability energy term
		*/
		double ECMaterialDurabilityEnergy(std::vector<float> _qtyVec);

        void initializeECMaterials();
		std::vector<ECMaterialComponentData> getECMaterialsVec(){return ECMaterialsVec;}
		std::vector<ECMaterialComponentData> *getECMaterialsVecPtr() { return &ECMaterialsVec; }
		unsigned int getNumberOfMaterials(){return numberOfMaterials;}
		std::map<std::string, int> getECMaterialNameIndexMap(){return ECMaterialNameIndexMap;}
		void setMaterialNameVector();
		std::vector<std::string> getMaterialNameVector() { 
			setMaterialNameVector();
			return fieldNameVec;
		}
		std::vector<float> checkQuantities(std::vector<float> _qtyVec);

		//functions used to manipulate extracellular material cell definitions
		void setRemodelingQuantityByName(const CellG * _cell, std::string _ECMaterialName, float _quantity);
		void setRemodelingQuantityByIndex(const CellG * _cell, int _idx, float _quantity);
		void setRemodelingQuantityVector(const CellG * _cell, std::vector<float> _quantityVec);
		void assignNewRemodelingQuantityVector(const CellG * _cell, int _numMtls=-1);
		//Medium functions
		void setMediumECMaterialQuantityByName(const Point3D &pt, std::string _ECMaterialName, float _quantity);
		void setMediumECMaterialQuantityByIndex(const Point3D &pt, int _idx, float _quantity);
		void setMediumECMaterialQuantityVector(const Point3D &pt, std::vector<float> _quantityVec);
		void assignNewMediumECMaterialQuantityVector(const Point3D &pt, int _numMtls);
		//material functions
		void setECMaterialDurabilityByName(std::string _ECMaterialName, float _durabilityLM);
		void setECMaterialDurabilityByIndex(int _idx, float _durabilityLM);
		void setECMaterialAdvectingByName(std::string _ECMaterialName, bool _isAdvecting);
		void setECMaterialAdvectingByIndex(int _idx, bool _isAdvecting);
		// adhesion functions
		void setECAdhesionByCell(const CellG *_cell, std::vector<float> _adhVec);

        //functions used to retrieve extracellular material cell definitions
		float getRemodelingQuantityByName(const CellG * _cell, std::string _ECMaterialName);
		float getRemodelingQuantityByIndex(const CellG * _cell, int _idx);
		std::vector<float> getRemodelingQuantityVector(const CellG * _cell);
		//Medium functions
		float getMediumECMaterialQuantityByName(const Point3D &pt, std::string _ECMaterialName);
		float getMediumECMaterialQuantityByIndex(const Point3D &pt, int _idx);
		std::vector<float> getMediumECMaterialQuantityVector(const Point3D &pt);
		std::vector<float> getMediumAdvectingECMaterialQuantityVector(const Point3D &pt);
		//material functions
		float getECMaterialDurabilityByName(std::string _ECMaterialName);
		float getECMaterialDurabilityByIndex(int _idx);
		bool getECMaterialAdvectingByName(std::string _ECMaterialName);
		bool getECMaterialAdvectingByIndex(int _idx);
		virtual int getECMaterialIndexByName(std::string _ECMaterialName);
		// adhesion functions
		std::vector<float> getECAdhesionByCell(const CellG *_cell);
		std::vector<float> getECAdhesionByCellTypeId(int _idx);

		//function to calculate material quantity vector when source is a medium site
		std::vector<float> calculateCopyQuantityVec(const CellG * _cell, const Point3D &pt);

		// drawing functions

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

		std::vector<Neighbor> getFirstOrderNeighbors(const Point3D &pt);

		virtual Field3D<ECMaterialsData *> *getECMaterialField() { return (Field3D<ECMaterialsData *> *)ECMaterialsField; }

		void overrideInitialization();

	};
}
#endif
