
#include <CompuCell3D/CC3D.h>        
using namespace CompuCell3D;

#include "NeighbourSurfaceConstraintPlugin.h"


NeighbourSurfaceConstraintPlugin::NeighbourSurfaceConstraintPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0)
{}

NeighbourSurfaceConstraintPlugin::~NeighbourSurfaceConstraintPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
}

void NeighbourSurfaceConstraintPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
   update(xmlData,true);
   
    
    potts->registerEnergyFunctionWithName(this,"NeighbourSurfaceConstraint");
        
    
    
    simulator->registerSteerableObject(this);
}

void NeighbourSurfaceConstraintPlugin::extraInit(Simulator *simulator){
    
}




double NeighbourSurfaceConstraintPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	

    double energy = 0;
    if (oldCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }
    
    if(newCell){
        //PUT YOUR CODE HERE
    }else{
        //PUT YOUR CODE HERE
    }
    
    return energy;    
}            


void NeighbourSurfaceConstraintPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
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


std::string NeighbourSurfaceConstraintPlugin::toString(){
    return "NeighbourSurfaceConstraint";
}


std::string NeighbourSurfaceConstraintPlugin::steerableName(){
    return toString();
}
