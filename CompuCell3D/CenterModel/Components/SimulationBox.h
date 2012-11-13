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

#ifndef SIMULATIONBOX_H
#define SIMULATIONBOX_H

#include "ComponentsDLLSpecifier.h"
#include <PublicUtilities/Vector3.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <set>
#include "CellCM.h"

namespace CenterModel {


	class COMPONENTS_EXPORT CellSorterDataCM{
	public:
		CellSorterDataCM(CellCM *_cell){cell=_cell;}
		CellCM * cell;

		bool operator<(const CellSorterDataCM & _rhs) const{
			return  cell->id < _rhs.cell->id;
		}

	};

	class COMPONENTS_EXPORT CellSorterCM{

	public:
		CellSorterCM(){}
	
		std::set<CellSorterDataCM> sorterSet;

	};

	class CompuCell3D::BoundaryStrategy;


	class InteractionRangeIterator;


	class COMPONENTS_EXPORT SimulationBox{
	public:

		
		typedef CompuCell3D::Field3DImpl<CellSorterCM * > LookupField_t;

		SimulationBox();

		virtual ~SimulationBox();

		void setDim(double _x=0,double _y=0,double _z=0) ;

		InteractionRangeIterator getInteractionRangeIterator(CellCM *_cell);

		void setGridSpacing(double _x=0,double _y=0,double _z=0);

		void setBoxSpatialProperties(double _x=0,double _y=0,double _z=0,double _xs=1.,double _ys=1.,double _zs=1.);

		void setBoxSpatialProperties(Vector3 & _dim, Vector3 & _gridSpacing);

		void setLookupLatticeDim(short _x,short _y, short _z);
		
		void updateCellLookup(CellCM * _cell);
		inline CompuCell3D::Point3D getCellLatticeLocation(const CellCM * _cell) const{return CompuCell3D::Point3D(static_cast<short>(floor(_cell->position.fX/gridSpacing.fX)),static_cast<short>(floor(_cell->position.fY/gridSpacing.fY)),static_cast<short>(floor(_cell->position.fZ/gridSpacing.fZ)));}

		Vector3 getDim() {return dim;}
		CompuCell3D::Dim3D getLatticeLookupDim(){return lookupLatticeDim;}

		const LookupField_t & getLookupFieldRef(){return *lookupLatticePtr;}

		std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> getLatticeLocationsWithinInteractingRange(CellCM *_cell);
		

	private:

		Vector3 dim;
		Vector3 gridSpacing;
		Vector3 inverseGridSpacing;
		CompuCell3D::Dim3D lookupLatticeDim;
		LookupField_t *lookupLatticePtr;
		CompuCell3D::BoundaryStrategy *boundaryStrategy;
		unsigned int maxNeighborIndex;	
		unsigned int maxNeighborOrder;	
		//this will have to be changed to vec of vecs for parallel implementation
		std::vector<CompuCell3D::Point3D> neighborsVec;


	};


	class COMPONENTS_EXPORT InteractionRangeIterator{
	public:
		friend class SimulationBox;	
		InteractionRangeIterator();		
		void initialize();
		CellCM * operator *() const;
		InteractionRangeIterator & operator ++();

		bool  operator ==(const InteractionRangeIterator & _rhs);

		bool  operator !=(const InteractionRangeIterator & _rhs);
		InteractionRangeIterator& begin();
		InteractionRangeIterator& end();

	private:
		SimulationBox * sbPtr;
		CellCM *cell;

		//std::set<CellSorterDataCM>::iterator sitrBegin;
		//std::set<CellSorterDataCM>::iterator sitrEnd;
		std::set<CellSorterDataCM>::iterator currentEnd;
		std::set<CellSorterDataCM>::iterator sitrCurrent;
		unsigned int counter;
		SimulationBox::LookupField_t *lookupFieldPtr;
		std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> neighborListPair;
		std::set<CellSorterDataCM> *currentSorterSetPtr;

	};




};
#endif
