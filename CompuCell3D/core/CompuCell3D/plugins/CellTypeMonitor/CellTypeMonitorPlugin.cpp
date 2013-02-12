
 #include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>

// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;

// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <PublicUtilities/StringUtils.h>
// // // #include <algorithm>

#include "CellTypeMonitorPlugin.h"


CellTypeMonitorPlugin::CellTypeMonitorPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0),
cellTypeArray(0),
mediumType(0)

{}

CellTypeMonitorPlugin::~CellTypeMonitorPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
    
    if (cellTypeArray){
        delete cellTypeArray;
        cellTypeArray=0;
    }
}

void CellTypeMonitorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
   update(xmlData,true);
   
    Dim3D fieldDim=cellFieldG->getDim();
    cellTypeArray=new Array3DCUDA<unsigned char>(fieldDim,mediumType);
    
    
    
    potts->registerCellGChangeWatcher(this);    
    
    
    simulator->registerSteerableObject(this);
}

void CellTypeMonitorPlugin::extraInit(Simulator *simulator){
    
}

            
void CellTypeMonitorPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 
{
    
    //This function will be called after each succesful pixel copy - field3DChange does usuall ohusekeeping tasks to make sure state of cells, and state of the lattice is uptdate
    // here we keep track of a cell type at a given position 
    if (newCell){
        cellTypeArray->set(pt,newCell->type);		
    }else{        
        cellTypeArray->set(pt,0);
    }

//     if (oldCell){
//         //PUT YOUR CODE HERE
//     }else{
//         //PUT YOUR CODE HERE
//     }
		
}




void CellTypeMonitorPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}


std::string CellTypeMonitorPlugin::toString(){
    return "CellTypeMonitor";
}


std::string CellTypeMonitorPlugin::steerableName(){
    return toString();
}
