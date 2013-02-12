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

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Simulator.h>

using namespace CompuCell3D;
using namespace std;

// // // #define EXP_STL
#include "VolumeTrackerPlugin.h"

VolumeTrackerPlugin::VolumeTrackerPlugin() : pUtils(0),lockPtr(0), potts(0), deadCellG(0) {
}

VolumeTrackerPlugin::~VolumeTrackerPlugin() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
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

