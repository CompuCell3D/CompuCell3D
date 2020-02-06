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

#include "PixelTrackerPlugin.h"


PixelTrackerPlugin::PixelTrackerPlugin():
simulator(0),potts(0)    
{
	fullInitAtStart = false;
	fullInitState = false;
	trackMedium = false;
}

PixelTrackerPlugin::~PixelTrackerPlugin() {}




void PixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


  simulator=_simulator;
  potts = simulator->getPotts();



  ///will register PixelTracker here
  BasicClassAccessorBase * cellPixelTrackerAccessorPtr=&pixelTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(cellPixelTrackerAccessorPtr);

  potts->registerCellGChangeWatcher(this);
  
  if (_xmlData) { 
	  trackMedium = _xmlData->findElement("TrackMedium");
	  fullInitAtStart = _xmlData->findElement("FullInitAtStart");
  }
  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	if (fullInitAtStart) {
		fullTrackerDataInit(pt, oldCell);
		fullInitAtStart = false;
	}

	if(newCell){
		std::set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;
		std::set<PixelTrackerData >::iterator sitr=pixelSetRef.find(PixelTrackerData(pt));
		pixelSetRef.insert(PixelTrackerData(pt));
	}
	else if (trackMedium) { mediumPixelSet.insert(PixelTrackerData(pt)); }

	std::set<PixelTrackerData >::iterator sitr;
	if(oldCell){
		std::set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
		sitr=pixelSetRef.find(PixelTrackerData(pt));

		ASSERT_OR_THROW("Could not find point:"+pt+" inside cell of id: "+BasicString(oldCell->id)+" type: "+BasicString((int)oldCell->type),
		sitr!=pixelSetRef.end());

		pixelSetRef.erase(sitr);
	}
	else if (trackMedium) {
		sitr = mediumPixelSet.find(PixelTrackerData(pt));

		ASSERT_OR_THROW("Could not find point:" + pt + " in medium.", sitr != mediumPixelSet.end());

		mediumPixelSet.erase(sitr);
	}
   
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PixelTrackerPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id!=LATTICE_RESIZE){
		return;
	}

	CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);

	Dim3D shiftVec=ev.shiftVec;

    CellInventory &cellInventory = potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG * cell;
        
    for(cInvItr=cellInventory.cellInventoryBegin() ; cInvItr !=cellInventory.cellInventoryEnd() ;++cInvItr )
    {
		cell=cInvItr->second;
		std::set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet;
		for (set<PixelTrackerData >::iterator sitr=pixelSetRef.begin() ; sitr != pixelSetRef.end() ; ++sitr ){
                        Point3D & pixel=const_cast<Point3D&>(sitr->pixel);
                        pixel.x+=shiftVec.x;
                        pixel.y+=shiftVec.y;
                        pixel.z+=shiftVec.z;
                        
// 			sitr->pixel.x+=shiftVec.x;
// 			sitr->pixel.y+=shiftVec.y;
// 			sitr->pixel.z+=shiftVec.z;
		}



    }

	if (trackMedium) { 

		for (set<PixelTrackerData >::iterator sitr = mediumPixelSet.begin(); sitr != mediumPixelSet.end(); ++sitr) {
			Point3D & pixel = const_cast<Point3D&>(sitr->pixel);
			pixel.x += shiftVec.x;
			pixel.y += shiftVec.y;
			pixel.z += shiftVec.z;
		}

	}

	fullInitState = false;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string PixelTrackerPlugin::toString(){
	return "PixelTracker";
}

void PixelTrackerPlugin::fullTrackerDataInit(Point3D ptChange, CellG *oldCell) {

	CellInventory &cellInventory = potts->getCellInventory();
	CellG *cell;
	for (CellInventory::cellInventoryIterator cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr) {
		cell = cInvItr->second;
		pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet.clear();
	}

	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *>*) potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();
	mediumPixelSet.clear();
	for (short z = 0; z < fieldDim.z; ++z)
		for (short y = 0; y < fieldDim.y; ++y)
			for (short x = 0; x < fieldDim.x; ++x) {
				Point3D pt = Point3D(x, y, z);
				if (pt == ptChange) { cell = oldCell; } else { cell = cellFieldG->get(pt); }
				if (cell) { pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet.insert(PixelTrackerData(pt)); }
				else if (trackMedium) { mediumPixelSet.insert(PixelTrackerData(pt)); }
			}

	fullInitState = true;

}

void PixelTrackerPlugin::fullMediumTrackerDataInit() {
	mediumPixelSet.clear();
	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *>*) potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();
	for (short z = 0; z < fieldDim.z; ++z)
		for (short y = 0; y < fieldDim.y; ++y)
			for (short x = 0; x < fieldDim.x; ++x) {
				Point3D pt = Point3D(x, y, z);
				CellG * cell = cellFieldG->get(pt);
				if (cell == 0) { mediumPixelSet.insert(PixelTrackerData(pt)); }
			}

}
