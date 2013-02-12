
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

#include "BoundaryMonitorPlugin.h"


BoundaryMonitorPlugin::BoundaryMonitorPlugin():
pUtils(0),
lockPtr(0),
xmlData(0) ,
cellFieldG(0),
boundaryStrategy(0),
maxNeighborIndex(0)
{}

BoundaryMonitorPlugin::~BoundaryMonitorPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr=0;
    
    if (boundaryArray){
        delete boundaryArray;
        boundaryArray=0;
    }    
    
}

void BoundaryMonitorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    
    pUtils=sim->getParallelUtils();
    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr); 
   
   update(xmlData,true);
   
    
    Dim3D fieldDim=cellFieldG->getDim();
	unsigned char initVal=0;
    boundaryArray=new Array3DCUDA<unsigned char>(fieldDim,initVal); // 0 indicates pixels is not a boundary pixel
    
    maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);    
    
    
    potts->registerCellGChangeWatcher(this);    
    
    
    simulator->registerSteerableObject(this);
}

void BoundaryMonitorPlugin::extraInit(Simulator *simulator){
    
}

            
void BoundaryMonitorPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) 
{
    
    if(newCell==oldCell){//happens during multiple calls to se fcn on the same pixel woth current cell - should be avoided
      return;
    }    
    //This function will be called after each succesful pixel copy - field3DChange does usuall ohusekeeping tasks to make sure state of cells, and state of the lattice is uptdate
    
    
    CellG *nCell;// neighbor cell
    CellG *nnCell; // neighbor of a neighbor cell
//     bool boundaryAssigned=false;
    Neighbor neighbor;
	Neighbor nneighbor;

    boundaryArray->set(pt,0);//assume that pt looses status of boundary pixel
    for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
        neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
        if(!neighbor.distance){
                //if distance is 0 then the neighbor returned is invalid
                continue;
        }

        nCell = cellFieldG->get(neighbor.pt);
        if (nCell!=newCell){ //if newPixel touches cell of different type this means it si a boundary pixel
            boundaryArray->set(pt,1);
//             break;
        }
        
        
        boundaryArray->set(neighbor.pt,0);//assume that neighbors of pt loose status of boundary pixels
        for(unsigned int nnIdx=0 ; nnIdx <= maxNeighborIndex ; ++nnIdx ){
            nneighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(neighbor.pt),nnIdx);
            if(!nneighbor.distance){
                    //if distance is 0 then the neighbor returned is invalid
                    continue;
            }
            
            nnCell = cellFieldG->get(nneighbor.pt);
             
            if (nneighbor.pt==pt){// after pixel copy pt will be occupied by new cell
                if(nCell!=newCell){
                    boundaryArray->set(neighbor.pt,1);//    
                    break;
                    
                }
//                 if(fieldG->get(nneighbor.pt)!=newCell){
//                     boundaryArray.set(neighbor.pt,1);//    
//                     break;
//                 }
                    
            }else{
                if(cellFieldG->get(nneighbor.pt)!=nCell){
                    boundaryArray->set(neighbor.pt,1);
                    break;    
                }
            }
            
        }
        
    }
    //if (!boundaryAssigned){
    //    boundaryArray->set(pt,0);
    //}

    ////dweling with pixels that ight loose status of boundary pixels
    //
    //
    //if (oldCell){
    //    CellG *nCell;
    //    bool boundaryAssigned=false;
    //    for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
    //        neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
    //        if(!neighbor.distance){
    //                //if distance is 0 then the neighbor returned is invalid
    //                continue;
    //        }

    //        nCell = fieldG->get(neighbor.pt);
    //        if (nCell!=newCell){ //if newPixel touches cell of different type this means it si a boundary pixel
    //            boundaryArray.set(pt,1);
    //            boundaryAssigned=true;
    //            break;
    //        }
    //        
    //    }        
    //    
    //    //PUT YOUR CODE HERE
    //}else{
    //    //PUT YOUR CODE HERE
    //}
		
}




void BoundaryMonitorPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
//     ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
//    set<unsigned char> cellTypesSet;

//     CC3DXMLElement * exampleXMLElem=_xmlData->getFirstElement("Example");
//     if (exampleXMLElem){
//         double param=exampleXMLElem->getDouble();
//         cerr<<"param="<<param<<endl;
//         if(exampleXMLElem->findAttribute("Type")){
//             std::string attrib=exampleXMLElem->getAttribute("Type");
//             // double attrib=exampleXMLElem->getAttributeAsDouble("Type"); //in case attribute is of type double
//             cerr<<"attrib="<<attrib<<endl;
//         }
//     }
    
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy=BoundaryStrategy::getInstance();

}


std::string BoundaryMonitorPlugin::toString(){
    return "BoundaryMonitor";
}


std::string BoundaryMonitorPlugin::steerableName(){
    return toString();
}
