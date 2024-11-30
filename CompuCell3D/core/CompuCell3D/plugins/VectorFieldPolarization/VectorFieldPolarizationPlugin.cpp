

#include <CompuCell3D/CC3D.h>        

using namespace CompuCell3D;

#include "VectorFieldPolarizationPlugin.h"

VectorFieldPolarizationPlugin::VectorFieldPolarizationPlugin():
        pUtils(nullptr),
        lockPtr(nullptr),
        xmlData(nullptr) ,
        cellFieldG(nullptr),
        boundaryStrategy(nullptr),
        vectorFieldPtr(nullptr)
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

    pUtils=sim->getParallelUtils();

    lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;

    pUtils->initLock(lockPtr);



    update(xmlData,true);
    // needs to be called after update method so  that we know the name of the field
    vectorFieldPtr = sim->createVectorField(vectorFieldName);
    

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
            }
}









double VectorFieldPolarizationPlugin::changeEnergy(const Point3D &pt,const CellG *newCell,const CellG *oldCell) {	


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

    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy=BoundaryStrategy::getInstance();

}

std::string VectorFieldPolarizationPlugin::toString(){
    return "VectorFieldPolarization";
}

std::string VectorFieldPolarizationPlugin::steerableName(){
    return toString();
}

