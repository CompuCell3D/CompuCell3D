
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
    double energy = 0;

    Point3D n;

    CellG *nCell=0;
    WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    Neighbor neighbor;

    if (oldCell == newCell) return 0;
    for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
    	neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
    	if(!neighbor.distance){
    		//if distance is 0 then the neighbor returned is invalid
    		continue;
    	}
    	nCell = fieldG->get(neighbor.pt);
    	if (nCell!=oldCell){
    		energy = 0; //place holder
    	}
    	if(nCell!=newCell){
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
    lambdaFaces.clear();
    targetFaces.clear();


    CC3DXMLElementList * faceLambdaVec = _xmlData->getElements("LambdaFace");
    CC3DXMLElementList * faceTargetVec = _xmlData->getElements("TargetFace");
    //I know this will work for stuff defined on the xml. I need to ask maciek about
    //doing it in python
    //This works for a symmetric things
    for (int i = 0 ; i<faceLambdaVec.size(); ++i){

    	setFaceLambda(faceLambdaVec[i]->getAtribute("Type1"),
    			faceLambdaVec[i]->getAtribute("Type2"), faceLambdaVec[i]->getDouble());

    	setFaceTarget(faceTargetVec[i]->getAtribute("Type1"),
    			faceTargetVec[i]->getAtribute("Type1"), faceTargetVec[i]->getDouble());

    	//inserting all the types to the set (duplicate are automatically eliminated)
    	// if it is the case this should be removed
    	//to figure out max value of type Id
    	cellTypesSet.insert(automaton->getTypeId(faceLambdaVec[i]->getAttribute("Type1")));
    	cellTypesSet.insert(automaton->getTypeId(faceLambdaVec[i]->getAttribute("Type2")));

    }
    //Now that we know all the types used in the simulation we will find size of
    //the lambdaFaceArray
    vector<unsigned char> cellTypesVector(cellTypesSet.begin(),
    		cellTypesSet.end());//coping set to the vector

    int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
    size+=1;//if max element is e.g. 5 then size has to be 6 for an
    //array to be properly allocated
    int index ;
    lambdaFacesArray.clear();
    targetFacesArray.clear();
    lambdaFacesArray.assign(size,vector<double>(size,0.0));
    targetFacesArray.assign(size,vector<double>(size,0.0));
    for(int i = 0 ; i < size ; ++i)
    		for(int j = 0 ; j < size ; ++j){

    			index = getIndex(cellTypesVector[i],cellTypesVector[j]);

    			lambdaFacesArray[i][j] = lambdaFaces[index];

    			targetFacesArray[i][j] = targetFaces[index];

	}

    cerr<<"size="<<size<<endl;

    for(int i = 0 ; i < size ; ++i)
    			for(int j = 0 ; j < size ; ++j){

    				cerr<<"lambdaFaces["<<i<<"]["<<j<<"]="<<lambdaFacesArray[i][j]<<endl;
    				cerr<<"targetFaces["<<i<<"]["<<j<<"]="<<targetFacesArray[i][j]<<endl;

	}















    /*
    if (exampleXMLElem){
        double param=exampleXMLElem->getDouble();
        cerr<<"param="<<param<<endl;
        if(exampleXMLElem->findAttribute("Type")){
            std::string attrib=exampleXMLElem->getAttribute("Type");
            // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
            cerr<<"attrib="<<attrib<<endl;
        }
    }
    */
    //check if there is a ScaleSurface parameter  in XML
	if(_xmlData->findElement("ScaleSurface")){
		scaleSurface=_xmlData->getFirstElement("ScaleSurface")->getDouble();
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
