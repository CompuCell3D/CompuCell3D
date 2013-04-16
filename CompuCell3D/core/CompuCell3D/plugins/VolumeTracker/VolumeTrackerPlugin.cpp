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

#include <CompuCell3D/CC3D.h>



using namespace CompuCell3D;
using namespace std;

#include "VolumeTrackerPlugin.h"

VolumeTrackerPlugin::VolumeTrackerPlugin() : pUtils(0),lockPtr(0), potts(0), deadCellG(0) {
}

VolumeTrackerPlugin::~VolumeTrackerPlugin() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
}

void VolumeTrackerPlugin::initVec(const vector<int> & _vec){
	cerr<<" THIS IS VEC.size="<<_vec.size()<<endl;
}

void VolumeTrackerPlugin::initVec(const Dim3D & _dim){
	cerr<<" THIS IS A COMPUCELL3D DIM3D"<<_dim<<endl;
}

bool VolumeTrackerPlugin::checkIfOKToResize(Dim3D _newSize,Dim3D _shiftVec){

	Field3DImpl<CellG*> *cellField=(Field3DImpl<CellG*> *)potts->getCellFieldG();
	Dim3D fieldDim=cellField->getDim();
	Point3D pt;
	Point3D shiftVec(_shiftVec.x,_shiftVec.y,_shiftVec.z);
	Point3D shiftedPt;
	CellG *cell;

	//cerr<<"_newSize="<<_newSize<<endl;
	//cerr<<"_shiftVec="<<_shiftVec<<endl;

	for (pt.x=0 ; pt.x<fieldDim.x ; ++pt.x)
		for (pt.y=0 ; pt.y<fieldDim.y ; ++pt.y)
			for (pt.z=0 ; pt.z<fieldDim.z ; ++pt.z){
				cell=cellField->get(pt);
				if(cell){
					shiftedPt=pt+shiftVec;

					if(shiftedPt.x<0 || shiftedPt.x>=_newSize.x || shiftedPt.y<0 || shiftedPt.y>=_newSize.y || shiftedPt.z<0 || shiftedPt.z>=_newSize.z){
						return false;
					}
				}
				
			}
	return true;
}



void VolumeTrackerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData)
{
	sim=simulator;
	pUtils=sim->getParallelUtils();
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);


	potts = simulator->getPotts();
	potts->registerCellGChangeWatcher(this);
	potts->registerStepper(this);

	deadCellVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), (CellG*)0);
}


void VolumeTrackerPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){
		CC3DEventChangeNumberOfWorkNodes ev = static_cast<CC3DEventChangeNumberOfWorkNodes&>(_event);
		deadCellVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), (CellG*)0);		
		cerr<<"VolumeTrackerPlugin::handleEvent="<<endl;
	}
}


std::string VolumeTrackerPlugin::toString(){return "VolumeTracker";}

std::string VolumeTrackerPlugin::steerableName(){return toString();}


void VolumeTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 
{
	//if (newCell)
	//   newCell->volume++;
	//
	//if (oldCell)
	//   if((--oldCell->volume) == 0)
	//      deadCellG = oldCell;


	
	if (newCell)
		newCell->volume++;

	if (oldCell)
		
		if((--oldCell->volume) == 0){
		
			deadCellVec[pUtils->getCurrentWorkNodeNumber()] = oldCell;
		}
		
	
}

//have to fix it
void VolumeTrackerPlugin::step() {

	
	// if (deadCellG) {
	//cerr<<"THIS IS VolumeTrackerPlugin removing cell "<<deadCellG<<" deadCellG->type="<<(int)deadCellG->type<<" deadCellG->id="<<deadCellG->id<<endl; 
	//   potts->destroyCellG(deadCellG);
	//   deadCellG = 0;
	// }
	CellG *deadCellPtr=deadCellVec[pUtils->getCurrentWorkNodeNumber()];
//         cerr<<"inside step function "<<deadCellPtr<<endl;
	if (deadCellPtr) {

        //NOTICE: we cannot use #pragma omp critical instead of locks because although this is called from inside the parallel region critical directive has to be included explicitely inside #pragma omp parallel section - and this has to be known at the compile time
		pUtils->setLock(lockPtr);
		//cerr<<"THIS IS VolumeTrackerPlugin removing cell "<<deadCellPtr<<" deadCellG->type="<<(int)deadCellPtr->type<<" deadCellG->id="<<deadCellPtr->id<<" thread="<<pUtils->getCurrentWorkNodeNumber()<<endl; 
		potts->destroyCellG(deadCellPtr);
		deadCellVec[pUtils->getCurrentWorkNodeNumber()]=0;
		pUtils->unsetLock(lockPtr);
	}

	
}

