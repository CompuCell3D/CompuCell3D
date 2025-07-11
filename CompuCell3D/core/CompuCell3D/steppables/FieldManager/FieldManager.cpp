

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

        fieldSpec.printInfo();

        if (fieldSpec.type == FieldSpec::FieldType::Vector){
            sim->createVectorField(fieldSpec.name);
        } else if (fieldSpec.type == FieldSpec::FieldType::Scalar){
            if (fieldSpec.kind == FieldSpec::FieldKind::CC3D){
                ASSERT_OR_THROW("Creation of legacy fields in FieldManager is not supported FieldManager Steppable", false)
            } else if (fieldSpec.kind == FieldSpec::FieldKind::NumPy){
                switch (fieldSpec.precision){
                    case FieldSpec::PrecisionType::Float:
                        sim->createGenericScalarField<float>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::Double:
                        sim->createGenericScalarField<double>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::Char:
                        sim->createGenericScalarField<char>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::UChar:
                        sim->createGenericScalarField<unsigned char>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::Short:
                        sim->createGenericScalarField<short>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::UShort:
                        sim->createGenericScalarField<unsigned short>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::Int:
                        sim->createGenericScalarField<int>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::UInt:
                        sim->createGenericScalarField<unsigned int>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::Long:
                        sim->createGenericScalarField<long>(fieldSpec.name, fieldSpec.padding);
                        break;
                    case FieldSpec::PrecisionType::ULong:
                        sim->createGenericScalarField<unsigned long>(fieldSpec.name, fieldSpec.padding);
                        break;
                    default:
                    ASSERT_OR_THROW("Unsupported precision type", false)
                }
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

        if (elem->findAttribute("Precision")){
            fieldSpec.precision = FieldSpec::mapStringToPrecision(elem->getAttribute("Precision"));
        } else {
            // default
            fieldSpec.precision = FieldSpec::PrecisionType::Float;
        }

        fieldSpecVec.push_back(fieldSpec);
    }


    //boundaryStrategy has information about pixel neighbors

    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string FieldManager::toString(){

   return "FieldManager";
}

std::string FieldManager::steerableName(){

   return toString();

}

