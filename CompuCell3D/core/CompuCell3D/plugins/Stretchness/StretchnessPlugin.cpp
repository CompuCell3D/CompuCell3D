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
 
#include <CompuCell3D/plugins/SurfaceTracker/SurfaceTrackerPlugin.h>

// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
using namespace CompuCell3D;



using namespace std;


#include "StretchnessPlugin.h"



StretchnessPlugin::StretchnessPlugin() : xmlData(0), cellFieldG(0),targetStretchness(0),lambdaStretchness(0),scaleSurface(1.0) {}

StretchnessPlugin::~StretchnessPlugin() {
  
}

void StretchnessPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  Potts3D *potts = simulator->getPotts();
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();


  potts->registerEnergyFunctionWithName(this,"StretchnessEnergy");
  simulator->registerSteerableObject(this);

  boundaryStrategy=BoundaryStrategy::getInstance();
 
   bool pluginAlreadyRegisteredFlag;
   SurfaceTrackerPlugin *trackerPlugin=(SurfaceTrackerPlugin*)Simulator::pluginManager.get("SurfaceTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
   if(!pluginAlreadyRegisteredFlag)
      trackerPlugin->init(simulator);

}

void StretchnessPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
   targetStretchness=_xmlData->getFirstElement("TargetStretchness")->getDouble();
   lambdaStretchness=_xmlData->getFirstElement("LambdaStretchness")->getDouble();
	if(_xmlData->findElement("ScaleSurface")){
		scaleSurface=_xmlData->getFirstElement("ScaleSurface")->getDouble();	
	}

	SurfaceTrackerPlugin *trackerPlugin=(SurfaceTrackerPlugin*)Simulator::pluginManager.get("SurfaceTracker"); 
   maxNeighborIndex=trackerPlugin->getMaxNeighborIndex() ;
   lmf= trackerPlugin->getLatticeMultiplicativeFactors();

}


void StretchnessPlugin::extraInit(Simulator *simulator) {
	
  update(xmlData,true);
}

double StretchnessPlugin::diffEnergy(double surface, double diff) {
  return lambdaStretchness *
    (diff*diff + 2 * diff * (surface - targetStretchness));
}


double StretchnessPlugin::changeEnergy(const Point3D &pt,
				  const CellG *newCell,
				  const CellG *oldCell) {

  // E = lambda * (surface - targetStretchness) ^ 2
  CellG *nCell;
  
  if (oldCell == newCell) return 0;

  unsigned int token = 0;
  double distance;
  int oldDiff = 0;
  int newDiff = 0;
  int newVolume=0;
  int oldVolume=0;
  float oldStretchness0=0.0;
  float newStretchness0=0.0;
  float oldStretchness1=0.0;
  float newStretchness1=0.0;
  float deltaStretchness=0.0;

  Point3D n;
  double energy = 0;

Log(LOG_TRACE) << "scaleStretchness="<<scaleStretchness<<" maxNeighborIndex="<<maxNeighborIndex;


  // Count surface difference

   Neighbor neighbor;
   for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
      neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
      if(!neighbor.distance){
      //if distance is 0 then the neighbor returned is invalid
      continue;
      }
      nCell = cellFieldG->get(neighbor.pt);
      if (newCell == nCell) newDiff-=lmf.surfaceMF;
      else newDiff+=lmf.surfaceMF;
   
      if (oldCell == nCell) oldDiff+=lmf.surfaceMF;
      else oldDiff-=lmf.surfaceMF;

   }

  if (newCell){
//       newVolume1=newCell->volume+1;
      newStretchness0=newCell->surface*scaleSurface/sqrt((float)newCell->volume);
      newStretchness1=(newCell->surface+newDiff)*scaleSurface/sqrt((float)newCell->volume+1);
      energy+=(newStretchness1-newStretchness0)*lambdaStretchness;

   }

  if (oldCell){
//       oldVolume1=oldCell->volume-1;
      oldStretchness0=oldCell->surface*scaleSurface/sqrt((float)oldCell->volume);
      
      oldStretchness1=(oldCell->volume > 1 ? 
                                          (oldCell->surface+oldDiff)*scaleSurface/sqrt((float)oldCell->volume-1) 
                                          :oldStretchness0
                     );
      energy+=(oldStretchness1-oldStretchness0)*lambdaStretchness;
   }

   
   


//   if (newCell){
//     energy += diffEnergy(newCell->surface*scaleSurface, newDiff*scaleSurface);
//    }
//   if (oldCell){
//     energy += diffEnergy(oldCell->surface*scaleSurface, oldDiff*scaleSurface);
//   }
   
Log(LOG_TRACE) << "energy="<<energy;
  return energy;
}



std::string StretchnessPlugin::toString(){
	return "Stretchness";
}


std::string StretchnessPlugin::steerableName(){
	return toString();
}


