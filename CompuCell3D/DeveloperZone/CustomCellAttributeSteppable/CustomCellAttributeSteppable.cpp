#include <CompuCell3D/CC3D.h>
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;
using namespace std;

#include "CustomCellAttributeSteppable.h"

CustomCellAttributeSteppable::CustomCellAttributeSteppable() : cellFieldG(0), sim(0), potts(0), xmlData(0), boundaryStrategy(0), automaton(0), cellInventoryPtr(0) {}



CustomCellAttributeSteppable::~CustomCellAttributeSteppable() {

}


void CustomCellAttributeSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;

    potts = simulator->getPotts();

    cellInventoryPtr = &potts->getCellInventory();

    sim = simulator;

    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

    fieldDim = cellFieldG->getDim();

    potts->getCellFactoryGroupPtr()->registerClass(&customCellAttributeSteppableDataAccessor);

    simulator->registerSteerableObject(this);

    update(_xmlData, true);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void CustomCellAttributeSteppable::extraInit(Simulator *simulator) {

    //PUT YOUR CODE HERE

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CustomCellAttributeSteppable::start() {



    //PUT YOUR CODE HERE



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CustomCellAttributeSteppableData * CustomCellAttributeSteppable::getCustomCellAttribute(CellG * cell) {

    CustomCellAttributeSteppableData * customCellAttrData = customCellAttributeSteppableDataAccessor.get(cell->extraAttribPtr);
    return customCellAttrData;
}


void CustomCellAttributeSteppable::step(const unsigned int currentStep) {

    //REPLACE SAMPLE CODE BELOW WITH YOUR OWN

    CellInventory::cellInventoryIterator cInvItr;

    CellG * cell = 0;

    
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr)

    {

        cell = cellInventoryPtr->getCell(cInvItr);
        
        CustomCellAttributeSteppableData * customCellAttrData = customCellAttributeSteppableDataAccessor.get(cell->extraAttribPtr);
        
        //storing cell id multiplied by currentStep in "x" member of the CustomCellAttributeSteppableData
        customCellAttrData->x = cell->id * currentStep;



        // storing last 5 xCOM positions in the "array" vector (part of  CustomCellAttributeSteppableData)
        std::vector<float> & vec = customCellAttrData->array;
        if (vec.size() < 5) {
            vec.push_back(cell->xCOM);
        }
        else
        {
            for (int i = 0; i < 4; ++i) {
                vec[i] = vec[i + 1];
            }
            vec[vec.size() - 1] = cell->xCOM;
        }


    }

    //printouts
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);
        CustomCellAttributeSteppableData * customCellAttrData = customCellAttributeSteppableDataAccessor.get(cell->extraAttribPtr);
        CC3D_Log(LOG_DEBUG) << "cell->id=" << cell->id << " mcs = " << currentStep << " attached x variable = " << customCellAttrData->x;
        CC3D_Log(LOG_DEBUG) << "----------- up to last 5 xCOM positions ----- for cell->id " << cell->id;
        for (int i = 0; i < customCellAttrData->array.size(); ++i) {
            CC3D_Log(LOG_DEBUG) << "x_com_pos[" << i << "]=" << customCellAttrData->array[i];
        }
    }

}


//std::vector<float>::iterator second_elem_itr = vec.begin();
//++second_elem_itr;
//vec.insert(vec.begin(), second_elem_itr, vec.end());


void CustomCellAttributeSteppable::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {



    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)


        //boundaryStrategy has information aobut pixel neighbors 
        boundaryStrategy = BoundaryStrategy::getInstance();

}



std::string CustomCellAttributeSteppable::toString() {

    return "CustomCellAttributeSteppable";

}



std::string CustomCellAttributeSteppable::steerableName() {

    return toString();

}



