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

	private:
		std::set<CellSorterDataCM> sorterSet;

	};

	class COMPONENTS_EXPORT SimulationBox{
	public:

		SimulationBox():lookupLatticePtr(0)
		{}

		virtual ~SimulationBox();

		void  setDim(double _x=0,double _y=0,double _z=0) ;

		void  setGridSpacing(double _x=0,double _y=0,double _z=0);

		void  setBoxSpatialProperties(double _x=0,double _y=0,double _z=0,double _xs=1.,double _ys=1.,double _zs=1.);

		void setLookupLatticeDim(short _x,short _y, short _z);


		Vector3 getDim() {return dim;}
		CompuCell3D::Dim3D getLatticeLookupDim(){return lookupLatticeDim;}

	private:

		Vector3 dim;
		Vector3 gridSpacing;
		CompuCell3D::Dim3D lookupLatticeDim;
		CompuCell3D::Field3DImpl<CellSorterCM * > *lookupLatticePtr;





	};



};
#endif
