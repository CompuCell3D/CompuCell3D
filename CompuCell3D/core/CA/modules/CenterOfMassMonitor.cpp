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

#include <CA/CACell.h> 
#include <CA/CAManager.h> 
#include <CompuCell3D/Field3D/WatchableField3D.h>

using namespace CompuCell3D;


using namespace std;


#include "CenterOfMassMonitor.h"
#include <iostream>

CenterOfMassMonitor::CenterOfMassMonitor():boundaryStrategy(0),caManager(0),cellField(0) {}

CenterOfMassMonitor::~CenterOfMassMonitor() {}


void CenterOfMassMonitor::init(CAManager *_caManager){
	RUNTIME_ASSERT_OR_THROW("CenterOfMassMonitor::init _caManager cannot be NULL!",_caManager);
	caManager=_caManager;
	cellField= caManager->getCellField();
	cerr<<"THIS IS COM cellField->getDim()="<<cellField->getDim()<<endl;

}

void CenterOfMassMonitor::field3DChange(const Point3D &pt, CACell *newCell,CACell * oldCell) {

	if (newCell){
		//when we move cell to a different location, in CA we set its previous site to NULL ptr because CA cell can only occupy one lattice site 
		//however we cannot do it here because of the infinite recursion problem...
		//if (newCell->xCOM >= 0 ){
		//	
		//	cellField->set(Point3D(newCell->xCOM,newCell->yCOM,newCell->zCOM),0); 
		//}
		newCell->xCOM = pt.x;
		newCell->yCOM = pt.y;
		newCell->zCOM = pt.z;
	}

    cerr<<"field 3d change center of mass monitor"<<endl;
	cerr<<"newCell="<<newCell<<endl;
	if (newCell){
		cerr<<"newCell COM=("<<newCell->xCOM<<","<<newCell->yCOM<<","<<newCell->zCOM<<")"<<endl;
	}

	cerr<<"oldCell="<<oldCell<<endl;
}


