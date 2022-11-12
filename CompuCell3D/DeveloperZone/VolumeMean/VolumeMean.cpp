



#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;


#include <CompuCell3D/Potts3D/CellInventory.h>

#include <iostream>
using namespace std;


#include "VolumeMean.h"

VolumeMean::VolumeMean() :  cellFieldG(0),sim(0),potts(0),exponent(1.0) {}

VolumeMean::~VolumeMean() {
}


void VolumeMean::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


  potts = simulator->getPotts();
  sim=simulator;
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();


  simulator->registerSteerableObject(this);
  update(_xmlData);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::extraInit(Simulator *simulator){

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VolumeMean::start(){

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::step(const unsigned int currentStep){

   CellInventory *cellInventoryPtr=& potts->getCellInventory();
   CellInventory::cellInventoryIterator cInvItr;
   CellG *cell;
   double mean=0.0;
   unsigned int cellCounter=0;
   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
      cell=cellInventoryPtr->getCell(cInvItr);
      // cell=*cInvItr;
      mean+=pow((double)cell->volume,(double)exponent);
      ++cellCounter;


   }
   if(cellCounter)
      mean/=cellCounter;
   else
      mean=0.0;
   CC3D_Log(LOG_DEBUG) << "The mean cell volume for exponent of "<<exponent<<" is "<<mean;

}






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VolumeMean::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

     if(_xmlData->findElement("Exponent"))
        exponent=_xmlData->getFirstElement("Exponent")->getDouble();

}

std::string VolumeMean::toString(){
   return "VolumeMean";
}


std::string VolumeMean::steerableName(){
   return toString();
}


