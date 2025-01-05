

#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "FieldManager.h"


FieldManager::FieldManager(): 
cellFieldG(0), sim(0), potts(0), 
xmlData(0), boundaryStrategy(0), automaton(0), cellInventoryPtr(0)
{}

FieldManager::~FieldManager() {
}

void FieldManager::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData=_xmlData;
    potts = simulator->getPotts();
    cellInventoryPtr=& potts->getCellInventory();
    sim=simulator;
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    fieldDim=cellFieldG->getDim();

    

    simulator->registerSteerableObject(this);

    update(_xmlData,true);

    //creating fields
    for (const auto& fieldSpec: fieldSpecVec){
        if (fieldSpec.type == FieldSpec::FieldType::Vector){
            sim->createVectorField(fieldSpec.name);
        }else if (fieldSpec.type == FieldSpec::FieldType::Scalar){
            if (fieldSpec.kind == FieldSpec::FieldKind::CC3D){
                // legacy scalar field -  probably nobody will ever use this option
                ASSERT_OR_THROW("Creation of legacy fields in FieldManager is not supported FieldManager Steppable", false)
            } else if (fieldSpec.kind == FieldSpec::FieldKind::NumPy){
                sim->createSharedNumpyConcentrationField(fieldSpec.name);

            }
        }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldManager::extraInit(Simulator *simulator){

    //PUT YOUR CODE HERE
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FieldManager::start(){

  //PUT YOUR CODE HERE
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FieldManager::step(const unsigned int currentStep){

}

void FieldManager::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

    fieldSpecVec.clear();
    CC3DXMLElementList fieldElemVec = _xmlData->getElements("Field");

    for (auto & elem: fieldElemVec){
        FieldSpec fieldSpec;
        fieldSpec.name = elem->getAttribute("Name");
        if (elem->findAttribute("Kind")){
            fieldSpec.kind = FieldSpec::mapStringToKind(elem->getAttribute("Kind"));
        }

        if (elem->findAttribute("Type")){
            fieldSpec.type = FieldSpec::mapStringToType(elem->getAttribute("Type"));
        }

        if (elem->findAttribute("Padding")){
            fieldSpec.padding = elem->getAttributeAsInt("Padding");
        }

        fieldSpecVec.push_back(fieldSpec);
    }


    //boundaryStrategy has information aobut pixel neighbors 

    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string FieldManager::toString(){

   return "FieldManager";
}

std::string FieldManager::steerableName(){

   return toString();

}

