

#include <CompuCell3D/CC3D.h>        
#include "CompuCell3D/Field3D/Field3DTypeBase.h"

using namespace CompuCell3D;

#include "VectorFieldPolarizationPlugin.h"

VectorFieldPolarizationPlugin::VectorFieldPolarizationPlugin():
        pUtils(nullptr),
        lockPtr(nullptr),
        xmlData(nullptr) ,
        cellFieldG(nullptr),
        boundaryStrategy(nullptr),
        vectorFieldPtr(nullptr),
        intNpyField(nullptr)
{}

VectorFieldPolarizationPlugin::~VectorFieldPolarizationPlugin() {

    pUtils->destroyLock(lockPtr);

    delete lockPtr;

    lockPtr= nullptr;

}

void VectorFieldPolarizationPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData=_xmlData;
    sim=simulator;
    potts=simulator->getPotts();
    cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
    fieldDim = cellFieldG->getDim();

    pUtils=sim->getParallelUtils();

    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;

    pUtils->initLock(lockPtr);



    update(xmlData,true);
    // needs to be called after update method so  that we know the name of the field
    vectorFieldPtr = sim->createVectorField(vectorFieldName);

    string intFieldName = "intNpyFieldCpp";
    sim->createGenericScalarField<int>(intFieldName);
    intNpyField = sim->getGenericScalarField<int>(intFieldName);

    potts->registerEnergyFunctionWithName(this,"VectorFieldPolarization");

    simulator->registerSteerableObject(this);

}

void VectorFieldPolarizationPlugin::extraInit(Simulator *simulator){
    Dim3D dim = cellFieldG->getDim();
    Point3D pt (0,0,0);
    for (pt.x=0 ; pt.x < dim.x ; ++pt.x)
        for (pt.y=0 ; pt.y < dim.y ; ++pt.y)
            for (pt.z=0 ; pt.z < dim.z ; ++pt.z){

                vectorFieldPtr->set(pt, vector<float>({static_cast<float>(pt.x), static_cast<float>(pt.y), 0}));
                intNpyField->set(pt, pt.x*pt.y);
            }
}


double VectorFieldPolarizationPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	
    // the energy term adds (negative) term to overall energy that enhances chances of accepting piuxel copy if
    // the displacement of the COM is along vector stored in the vector field


    double energy = 0;

    auto new_old_distance_vec_pair = proposedInvariantCOMShiftDueToPixelCopy(pt, newCell, oldCell, fieldDim, boundaryStrategy);


    if (oldCell){
        Point3D comPt((int) round(oldCell->xCM / oldCell->volume), (int) round(oldCell->yCM / oldCell->volume),
                      (int) round(oldCell->zCM / oldCell->volume));
        Coordinates3D<float> fiberVec = vectorFieldPtr->get(comPt);
        Coordinates3D<double> fiberVecDouble(fiberVec.X(), fiberVec.Y(), fiberVec.Z());
        Coordinates3D<double> & displacementVecOldCell = new_old_distance_vec_pair.second.coordinates3D;

        energy +=  -std::abs(polarizationLambda * (fiberVecDouble * displacementVecOldCell) );

    }


    if(newCell){
        Point3D comPt((int) round(newCell->xCM / newCell->volume), (int) round(newCell->yCM / newCell->volume),
                      (int) round(newCell->zCM / newCell->volume));
        Coordinates3D<float> fiberVec = vectorFieldPtr->get(comPt);
        Coordinates3D<double> fiberVecDouble(fiberVec.X(), fiberVec.Y(), fiberVec.Z());
        Coordinates3D<double> & displacementVecNewCell = new_old_distance_vec_pair.first.coordinates3D;

        energy +=  -std::abs(polarizationLambda * (fiberVecDouble * displacementVecNewCell) );

    }


    return energy;    
}            



void VectorFieldPolarizationPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

    //PARSE XML IN THIS FUNCTION

    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. "
                    "MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)

   set<unsigned char> cellTypesSet;

    CC3DXMLElement * fieldXMLElem=_xmlData->getFirstElement("Field");

    if (fieldXMLElem){


        if(fieldXMLElem->findAttribute("Name")){

            vectorFieldName = fieldXMLElem->getAttribute("Name");

        }

    }

    CC3DXMLElement * polarizationLambdaXMLElem = _xmlData->getFirstElement("PolarizationLambda");

    ASSERT_OR_THROW("You need to provide PolarizationLambda for VectorFieldPolarization", polarizationLambdaXMLElem)
    polarizationLambda = polarizationLambdaXMLElem->getDouble();

    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string VectorFieldPolarizationPlugin::toString(){
    return "VectorFieldPolarization";
}

std::string VectorFieldPolarizationPlugin::steerableName(){
    return toString();
}

