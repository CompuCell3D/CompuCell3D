#include <CompuCell3D/CC3D.h>
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;
using namespace std;

#include "GrowthSteppable.h"


GrowthSteppable::GrowthSteppable() : 
cellFieldG(0),sim(0),potts(0),xmlData(0),
boundaryStrategy(0),automaton(0),cellInventoryPtr(0){}

GrowthSteppable::~GrowthSteppable() {

}

void GrowthSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;

  potts = simulator->getPotts();

  cellInventoryPtr=& potts->getCellInventory();

  sim=simulator;

  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  fieldDim=cellFieldG->getDim();

  simulator->registerSteerableObject(this);

  update(_xmlData,true);
}

void GrowthSteppable::extraInit(Simulator *simulator){

}

void GrowthSteppable::start(){

    CellInventory::cellInventoryIterator cInvItr;
    CellG * cell = 0;

    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)
    {

        cell = cellInventoryPtr->getCell(cInvItr);
        cell->targetVolume = 25.0;
        cell->lambdaVolume = 2.0;

    }

}

void GrowthSteppable::step(const unsigned int currentStep){

    CellInventory::cellInventoryIterator cInvItr;

    CellG * cell=0;

   if (currentStep > 100)
       return;

    std::map<unsigned int, double>::iterator mitr;
        
    for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
    {

        cell=cellInventoryPtr->getCell(cInvItr);

        mitr = this->growthRateMap.find((unsigned int)cell->type);
        
        if (mitr != this->growthRateMap.end()){
            cell->targetVolume += mitr->second;
        }
        
    }

}

void GrowthSteppable::setGrowthRate(unsigned int cellType, double growthRate){
    CC3D_Log(LOG_DEBUG) << "CHANGING GROWTH RATE FOR CELL TYPE "<<cellType<<" TO "<<growthRate;
    std::map<unsigned int, double>::iterator mitr;
    this->growthRateMap[cellType] = growthRate;
}

void GrowthSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

    set<unsigned char> cellTypesSet;

    CC3DXMLElementList growthVec = _xmlData->getElements("GrowthRate");

    for (int i = 0; i < growthVec.size(); ++i) {
        unsigned int cellType = growthVec[i]->getAttributeAsUInt("CellType");
        double growthRateTmp = growthVec[i]->getAttributeAsDouble("Rate");
        this->growthRateMap[cellType] = growthRateTmp;
    }


    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string GrowthSteppable::toString(){

   return "GrowthSteppable";
}

std::string GrowthSteppable::steerableName(){

   return toString();
}

        

