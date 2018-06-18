
#include <CompuCell3D/CC3D.h>        
using namespace CompuCell3D;

#include "NeighborSurfaceConstraintPlugin.h"

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

NeighborSurfaceConstraintPlugin::NeighborSurfaceConstraintPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0)
{}

NeighborSurfaceConstraintPlugin::~NeighborSurfaceConstraintPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
}

void NeighborSurfaceConstraintPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
   update(xmlData,true);
    potts->registerEnergyFunctionWithName(this,"NeighborSurfaceConstraint");
        
    
    
    simulator->registerSteerableObject(this);
}

void NeighborSurfaceConstraintPlugin::extraInit(Simulator *simulator){
    
}
// energy will be defined on the types of cells. So something like the contact energy will be needed -> this will be more complicated, done later
// need to get neighbor data (Area)
// on neighbor stick there is some code that may be relevant
/*
 * Neighbor neighbor;   //Used by NeighborFinder to hold the offset to a neighbor Point3D and it's distance.
 * std::set<NeighborSurfaceData> * neighborData;
 * std::set<NeighborSurfaceData >::iterator sitr;
 *
 * neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors); // actually gets the data
 */


// need something that will calculate the surface difference only with this one
// neighbor.
// I believe this will do it
std::pair<double,double> NeighborSurfaceConstraintPlugin::getNewOldSurfaceDiffs(const Point3D &pt, const CellG *newCell,const CellG *oldCell){


	CellG *nCell;
   double oldDiff = 0.;
   double newDiff = 0.;
   Neighbor neighbor;
   for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
      neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
      if(!neighbor.distance){
      //if distance is 0 then the neighbor returned is invalid
      continue;
      }
      nCell = cellFieldG->get(neighbor.pt);
      if (newCell == nCell) newDiff-=lmf.surfaceMF;

      if (oldCell == nCell) oldDiff+=lmf.surfaceMF;

   }
	return make_pair(newDiff,oldDiff);
}



//energy difference function
double NeighborSurfaceConstraintPlugin::energyChange(double lambda, double targetSurface,double surface,  double diff) {
	if (!energyExpressionDefined){
		return lambda *(diff*diff + 2 * diff * (surface - fabs(targetSurface)));
	}
	else{
		return 0;//place holder
	}
}



double NeighborSurfaceConstraintPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	
	// This plugin does not make sense if the user is not using it by at least cell type.
	std::set<NeighborSurfaceData> * neighborData;
	std::set<NeighborSurfaceData >::iterator sitr;
    double energy = 0;
    if (oldCell == newCell) return 0;
    neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
    for(sitr=neighborData->begin() ; sitr !=neighborData ->end() ; ++sitr ){


    	if (oldCell){

    		energy = 0; //place holder
    	}
    	if(newCell){
    		energy = 0; //place holder
    	}
    }
    return energy;    
}            


void NeighborSurfaceConstraintPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
   set<unsigned char> cellTypesSet;

    CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");
    if (exampleXMLElem){
        double param=exampleXMLElem->getDouble();
        cerr<<"param="<<param<<endl;
        if(exampleXMLElem->findAttribute("Type")){
            std::string attrib=exampleXMLElem->getAttribute("Type");
            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
            cerr<<"attrib="<<attrib<<endl;
        }
    }
    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}


std::string NeighborSurfaceConstraintPlugin::toString(){
    return "NeighborSurfaceConstraint";
}


std::string NeighborSurfaceConstraintPlugin::steerableName(){
    return toString();
}
