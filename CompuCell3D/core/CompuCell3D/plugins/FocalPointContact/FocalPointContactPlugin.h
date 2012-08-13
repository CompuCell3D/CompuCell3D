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

#ifndef FOCALPOINTCONTACTPLUGIN_H
#define FOCALPOINTCONTACTPLUGIN_H


#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <CompuCell3D/Plugin.h>
#include "FocalPointBoundaryPixelTracker.h"

#include <map>
#include <vector>


#include "FocalPointContactDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
	class Potts3D;
	class Automaton;
	class BoundaryStrategy;
    class ParallelUtilsOpenMP;

	class FOCALPOINTCONTACT_EXPORT FocalPointContactPlugin : public Plugin,public EnergyFunction, public CellGChangeWatcher  {

      
      BasicClassAccessor<FocalPointBoundaryPixelTracker> focalPointBoundaryPixelTrackerAccessor;
      
        ParallelUtilsOpenMP * pUtils;
		Potts3D *potts;

        
		typedef std::map<int, double> contactEnergies_t;
		typedef std::vector<std::vector<double> > contactEnergyArray_t;

		contactEnergies_t contactEnergies;

		contactEnergyArray_t contactEnergyArray;

		std::string autoName;
		double depth;

		Automaton *automaton;
		bool weightDistance;
		unsigned int maxNeighborIndex;
	   unsigned int maxNeighborIndexJunctionMove;
		BoundaryStrategy * boundaryStrategy;
		CC3DXMLElement *xmlData;
      
      


      
      double lambda;
	  double lambdaSpring;
      double offset;
      double targetDistance;
      double potentialFunction(double _lambda,double _offset,double _targetDistance, double _distance);
      
      //vectorized variables for easy parallel access
      std::vector<short> returnedJunctionToPoolFlagVec;
      std::vector<short> newJunctionInitiatedFlagVec;      
      std::vector<FocalPointBoundaryPixelTrackerData> fpbPixDataOldCellBeforeVec;
      std::vector<FocalPointBoundaryPixelTrackerData> fpbPixDataOldCellAfterVec;
      std::vector<FocalPointBoundaryPixelTrackerData> fpbPixDataNewCellBeforeVec;
      std::vector<FocalPointBoundaryPixelTrackerData> fpbPixDataNewCellAfterVec;

      
		unsigned int maxNumberOfJunctions;


	public:
		FocalPointContactPlugin();
		virtual ~FocalPointContactPlugin();
		//Plugin interface
		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator);
		
		

		//EnergyFunction Interface
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);
		double changeEnergySprings(const Point3D &pt,const CellG *newCell,const CellG *oldCell);


      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);




		double contactEnergy(const CellG *cell1, const CellG *cell2);

		/**
		* Sets the contact energy for two cell types.  A -1 type is interpreted
		* as the medium.
		*/
		void setContactEnergy(const std::string typeName1,
			const std::string typeName2, const double energy);


		//Steerable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();
	protected:
		/**
		* @return The index used for ordering contact energies in the map.
		*/
		int getIndex(const int type1, const int type2) const;

	};
};
#endif
