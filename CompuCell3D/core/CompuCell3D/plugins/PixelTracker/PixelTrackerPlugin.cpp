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
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

using namespace CompuCell3D;


// // // #include <iostream>
// // // #include <cmath>
using namespace std;

#include "PixelTrackerPlugin.h"


PixelTrackerPlugin::PixelTrackerPlugin():
simulator(0)    
{}

PixelTrackerPlugin::~PixelTrackerPlugin() {}




void PixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


  simulator=_simulator;
  Potts3D *potts = simulator->getPotts();



  ///will register PixelTracker here
  BasicClassAccessorBase * cellPixelTrackerAccessorPtr=&pixelTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(cellPixelTrackerAccessorPtr);

  potts->registerCellGChangeWatcher(this);
  


}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	if(newCell){
		std::set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;
		std::set<PixelTrackerData >::iterator sitr=pixelSetRef.find(PixelTrackerData(pt));
		pixelSetRef.insert(PixelTrackerData(pt));
	}

	if(oldCell){
		std::set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
		std::set<PixelTrackerData >::iterator sitr;
		sitr=pixelSetRef.find(PixelTrackerData(pt));

		ASSERT_OR_THROW("Could not find point:"+pt+" inside cell of id: "+BasicString(oldCell->id)+" type: "+BasicString((int)oldCell->type),
		sitr!=pixelSetRef.end());

		pixelSetRef.erase(sitr);
	}

   
   
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string PixelTrackerPlugin::toString(){
	return "PixelTracker";
}
