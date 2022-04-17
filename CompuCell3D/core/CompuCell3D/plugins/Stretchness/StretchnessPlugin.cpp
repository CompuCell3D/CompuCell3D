

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


//    cerr<<"scaleStretchness="<<scaleStretchness<<" maxNeighborIndex="<<maxNeighborIndex<<endl;


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
   

//   cerr<<"energy="<<energy<<endl;
  return energy;
}



std::string StretchnessPlugin::toString(){
	return "Stretchness";
}


std::string StretchnessPlugin::steerableName(){
	return toString();
}


