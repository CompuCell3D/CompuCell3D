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

#include "SimulationBox.h"
#include <iostream>
#include <cmath>

//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * lookupLatticeDim.y)) * lookupLatticeDim.x))

using namespace std;
using namespace CenterModel;


SimulationBox::~SimulationBox(){

	if (!lookupLatticePtr)
		return;
	
	
	CompuCell3D::Point3D pt;
	for (pt.x = 0 ; pt.x < lookupLatticeDim.x ; ++pt.x)
		for (pt.y = 0 ; pt.y < lookupLatticeDim.y ; ++pt.y)
			for (pt.z = 0 ; pt.z < lookupLatticeDim.z ; ++pt.z){

				lookupLatticePtr->set(pt,new CellSorterCM());
			}

	delete lookupLatticePtr;


}

void  SimulationBox::setDim(double _x,double _y,double _z) {
	//dim.fX=(ceil(fabs(_x)));
	//dim.fY=(ceil(fabs(_y)));
	//dim.fZ=(ceil(fabs(_z)));

	dim.fX=_x;
	dim.fY=_y;
	dim.fZ=_z;

}


void SimulationBox::setGridSpacing(double _x,double _y,double _z){
	//gridSpacing.fX=(ceil(fabs(_x)));
	//gridSpacing.fY=(ceil(fabs(_y)));
	//gridSpacing.fZ=(ceil(fabs(_z)));

	gridSpacing.fX=_x;
	gridSpacing.fY=_y;
	gridSpacing.fZ=_z;

}

void SimulationBox::setBoxSpatialProperties(Vector3 & _dim, Vector3 & _gridSpacing){
	setBoxSpatialProperties(_dim.fX,_dim.fY,_dim.fZ,_gridSpacing.fX,_gridSpacing.fY,_gridSpacing.fZ);	
}

void SimulationBox::setBoxSpatialProperties(double _x,double _y,double _z,double _xs,double _ys,double _zs){

	dim.fX=_x;
	dim.fY=_y;
	dim.fZ=_z;

	gridSpacing.fX=_xs;
	gridSpacing.fY=_ys;
	gridSpacing.fZ=_zs;

	inverseGridSpacing.fX=1.0/gridSpacing.fX;
	inverseGridSpacing.fY=1.0/gridSpacing.fY;
	inverseGridSpacing.fZ=1.0/gridSpacing.fZ;

	double xratio=1.0,yratio=1.0,zratio=1.0;

	xratio=dim.fX/gridSpacing.fX;
	yratio=dim.fY/gridSpacing.fY;
	zratio=dim.fZ/gridSpacing.fZ;
	//cerr<<"xratio="<<xratio<<endl;
	//cerr<<"yratio="<<yratio<<endl;
	//cerr<<"zratio="<<zratio<<endl;


	lookupLatticeDim.x=static_cast<short>(floor(fabs(xratio)))+1;
	lookupLatticeDim.y=static_cast<short>(floor(fabs(yratio)))+1;
	lookupLatticeDim.z=static_cast<short>(floor(fabs(zratio)))+1;


	lookupLatticePtr=new CompuCell3D::Field3DImpl<CellSorterCM*>(lookupLatticeDim,static_cast<CellSorterCM*>(0));


	CompuCell3D::Point3D pt;
	for (pt.x = 0 ; pt.x < lookupLatticeDim.x ; ++pt.x)
		for (pt.y = 0 ; pt.y < lookupLatticeDim.y ; ++pt.y)
			for (pt.z = 0 ; pt.z < lookupLatticeDim.z ; ++pt.z){

				lookupLatticePtr->set(pt,new CellSorterCM());
			}


			//for (int i = 0 ; i < lookupLatticeDim.x ; ++i){
			//	pt.x=x;

			//	lookupLatticePtr->get(pt)
			//}


}


void SimulationBox::setLookupLatticeDim(short _x,short _y, short _z){

	lookupLatticeDim.x=_x;
	lookupLatticeDim.y=_y;
	lookupLatticeDim.z=_z;

}

void SimulationBox::updateCellLookup(CellCM * _cell){
	
	//Vector3 lookupPosition=_cell->position*inverseGridSpacing;

	CompuCell3D::Point3D pt(static_cast<short>(floor(_cell->position.fX/gridSpacing.fX)),static_cast<short>(floor(_cell->position.fY/gridSpacing.fY)),static_cast<short>(floor(_cell->position.fZ/gridSpacing.fZ)));

	//cerr<<"_cell->position="<<_cell->position<<endl;
	//cerr<<"inverseGridSpacing="<<inverseGridSpacing<<endl;
	//cerr<<"lookupPosition="<<lookupPosition<<endl;

	//cerr<<"pt="<<pt<<endl;	
	long newLookupIndex=PT2IDX(pt);
	long oldLookupIndex=_cell->lookupIdx;
	if (oldLookupIndex!=newLookupIndex){
		//fetch lookup set  corresponding to _cell->lookupIdx
		
		CellSorterCM * csPtr=lookupLatticePtr->getByIndex(oldLookupIndex);
		if (csPtr){
			//making sure that cell's lookup index is sane 
			//cerr<<"csPtr="<<csPtr<<endl;

			set<CellSorterDataCM> & oldSorterSetRef=lookupLatticePtr->getByIndex(oldLookupIndex)->sorterSet;
			
			//cerr<<"oldSorterSetRef.size()="<<oldSorterSetRef.size()<<endl;		

			set<CellSorterDataCM>::iterator oldSitr=oldSorterSetRef.find(CellSorterDataCM(_cell));

			

			if(oldSitr!=oldSorterSetRef.end()){//cell is in the location pointed by lookupIdx and we remove it
				oldSorterSetRef.erase(oldSitr);
			}

		}

		//now insert cell with updated lookup index ()
		_cell->lookupIdx=newLookupIndex;
		//cerr<<"newLookupIndex="<<newLookupIndex<<endl;
		set<CellSorterDataCM> & newSorterSetRef=lookupLatticePtr->getByIndex(newLookupIndex)->sorterSet;
		set<CellSorterDataCM>::iterator newSitr;
		newSorterSetRef.insert(CellSorterDataCM(_cell));
		//cerr<<"newSorterSetRef.size()="<<newSorterSetRef.size()<<endl;

	}

	
	
}