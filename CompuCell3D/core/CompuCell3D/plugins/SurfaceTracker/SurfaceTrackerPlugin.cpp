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
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
using namespace CompuCell3D;



using namespace std;


#include "SurfaceTrackerPlugin.h"

SurfaceTrackerPlugin::SurfaceTrackerPlugin() :  cellFieldG(0),boundaryStrategy(0),maxNeighborIndex(0) {}

SurfaceTrackerPlugin::~SurfaceTrackerPlugin() {
}



void SurfaceTrackerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  potts = simulator->getPotts();
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
  
  potts->registerCellGChangeWatcher(this);

  boundaryStrategy=BoundaryStrategy::getInstance();

   update(_xmlData);
   
   simulator->registerSteerableObject(this);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SurfaceTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 
{
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;

	unsigned int token 	= 0;
	double distance;
	double oldDiff 		= 0.;
	double newDiff 		= 0.;
	CellG *nCell		= 0;
	Neighbor neighbor; 

   for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx )
	{
		neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance)
			continue;

		nCell = cellFieldG->get(neighbor.pt);
		if (newCell == nCell) newDiff-=lmf.surfaceMF;
		else newDiff+=lmf.surfaceMF;
		
		if (oldCell == nCell) oldDiff+=lmf.surfaceMF;
		else oldDiff-=lmf.surfaceMF;
   }

  if (newCell) newCell->surface += newDiff;
  if (oldCell) oldCell->surface += oldDiff;
}




void SurfaceTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	if(!_xmlData){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1); //use first nearest neighbors for surface calculations as default
	}else if(_xmlData->getFirstElement("MaxNeighborOrder")){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("MaxNeighborOrder")->getUInt());
	}else if (_xmlData->getFirstElement("MaxNeighborDistance")){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("MaxNeighborDistance")->getDouble());//depth=1.1 - means 1st nearest neighbor
	}else{
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1); //use first nearest neighbors for surface calculations as default
	}
	lmf=boundaryStrategy->getLatticeMultiplicativeFactors();
}

std::string SurfaceTrackerPlugin::steerableName(){return toString();}

std::string SurfaceTrackerPlugin::toString(){return "SurfaceTracker";}