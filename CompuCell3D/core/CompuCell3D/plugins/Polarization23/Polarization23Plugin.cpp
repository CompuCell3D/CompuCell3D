#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "Polarization23Plugin.h"


Polarization23Plugin::Polarization23Plugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        boundaryStrategy(0) {}

Polarization23Plugin::~Polarization23Plugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void Polarization23Plugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    fieldDim = cellFieldG->getDim();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    update(xmlData, true);

    potts->getCellFactoryGroupPtr()->registerClass(&polarization23DataAccessor);
    potts->registerEnergyFunctionWithName(this, "Polarization23");


    simulator->registerSteerableObject(this);
}

void Polarization23Plugin::extraInit(Simulator *simulator) {

}

double Polarization23Plugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    if (oldCell == newCell) return 0.0;
    Polarization23Data oldPolData;
    Polarization23Data newPolData;

    Vector3 oldPolVec;
    Vector3 newPolVec;


    Vector3 oldCOMAfter;
    Vector3 newCOMAfter;

    Vector3 oldPolVecBefore;
    Vector3 newPolVecBefore;

    Vector3 oldPolVecAfter;
    Vector3 newPolVecAfter;

    CellG *newComp1 = 0, *newComp2 = 0;
    CellG *oldComp1 = 0, *oldComp2 = 0;
    bool oldCellAboutToDisappear = false;


    if (oldCell) {
//         oldPolVec=polarization23DataAccessor.get(oldCell->extraAttribPtr)->polarizationVec;        
        oldPolData = *polarization23DataAccessor.get(oldCell->extraAttribPtr);
        oldPolVec = oldPolData.polarizationVec;
    }

    if (newCell) {
//         newPolVec=polarization23DataAccessor.get(newCell->extraAttribPtr)->polarizationVec;        
        newPolData = *polarization23DataAccessor.get(newCell->extraAttribPtr);
        newPolVec = newPolData.polarizationVec;

    }

    double energy = 0.0;
    if (oldCell) {

        CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(oldCell->clusterId);

        double clusterSurface;
        for (int i = 0; i < compartments.size(); ++i) {
            if (!oldComp1 && compartments[i]->type == oldPolData.type1) {
                oldComp1 = compartments[i];
            }

            if (compartments[i]->type == oldPolData.type2) {
                oldComp2 = compartments[i];
            }

        }

        if (oldComp1 && oldComp2) {

            Coordinates3D<double> oldPolVecBeforeCoordinates = distanceVectorCoordinatesInvariant(
                    Coordinates3D<double>(oldComp1->xCOM, oldComp1->yCOM, oldComp1->zCOM),
                    Coordinates3D<double>(oldComp2->xCOM, oldComp2->yCOM, oldComp2->zCOM), fieldDim);
            oldPolVecBefore.fX = oldPolVecBeforeCoordinates.X();
            oldPolVecBefore.fY = oldPolVecBeforeCoordinates.Y();
            oldPolVecBefore.fZ = oldPolVecBeforeCoordinates.Z();
        }

        Coordinates3D<double> centroidOldAfter;

        if (oldCell->volume > 1) {
            centroidOldAfter = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);
            oldCOMAfter.fX = centroidOldAfter.X() / (float) (oldCell->volume - 1);
            oldCOMAfter.fY = centroidOldAfter.Y() / (float) (oldCell->volume - 1);
            oldCOMAfter.fZ = centroidOldAfter.Z() / (float) (oldCell->volume - 1);


        } else {

            //if cell is about to disappear we do not impose any energy penalty in this plugin            

            oldCellAboutToDisappear = true;
        }

    }

    /////////////////////////////////////////

    if (newCell) {

        CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(newCell->clusterId);

        double clusterSurface;
        for (int i = 0; i < compartments.size(); ++i) {
            if (!newComp1 && compartments[i]->type == newPolData.type1) {
                newComp1 = compartments[i];
            }

            if (compartments[i]->type == newPolData.type2) {
                newComp2 = compartments[i];
            }

        }

        if (newComp1 && newComp2) {

            Coordinates3D<double> newPolVecBeforeCoordinates = distanceVectorCoordinatesInvariant(
                    Coordinates3D<double>(newComp1->xCOM, newComp1->yCOM, newComp1->zCOM),
                    Coordinates3D<double>(newComp2->xCOM, newComp2->yCOM, newComp2->zCOM), fieldDim);
            newPolVecBefore.fX = newPolVecBeforeCoordinates.X();
            newPolVecBefore.fY = newPolVecBeforeCoordinates.Y();
            newPolVecBefore.fZ = newPolVecBeforeCoordinates.Z();
        }


        Coordinates3D<double> centroidNewAfter = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);
        newCOMAfter.fX = centroidNewAfter.X() / (float) (newCell->volume + 1);
        newCOMAfter.fY = centroidNewAfter.Y() / (float) (newCell->volume + 1);
        newCOMAfter.fZ = centroidNewAfter.Z() / (float) (newCell->volume + 1);

    }


    //calculate position of polarization vector for cluster of oldCell after the flip
    if (oldCell) {
        bool oldCellInvolved = false;//involved in polarization vector calculations
        if (oldComp1 && oldComp2) {
            Vector3 vec1; //corresponds to oldComp1 after flip
            Vector3 vec2; //corresponds to oldComp2 after flip
            if (oldComp1 == oldCell) {
                vec1 = oldCOMAfter;
                oldCellInvolved = true;
            } else if (oldComp1 == newCell) {
                vec1 = newCOMAfter;
            } else {
                vec1.fX = oldComp1->xCOM;
                vec1.fY = oldComp1->yCOM;
                vec1.fZ = oldComp1->zCOM;
            }

            if (oldComp2 == oldCell) {
                vec2 = oldCOMAfter;
                oldCellInvolved = true;
            } else if (oldComp2 == newCell) {
                vec2 = newCOMAfter;
            } else {
                vec2.fX = oldComp2->xCOM;
                vec2.fY = oldComp2->yCOM;
                vec2.fZ = oldComp2->zCOM;
            }

            if (oldCellAboutToDisappear && oldCellInvolved) {
//                 oldPolVecAfter=Vector3(0.0,0.0,0.0);
                //if cell is about to disappear we do not impose any energy penalty in this plugin            
                oldPolVecAfter = oldPolVecBefore;

            } else {
                Coordinates3D<double> oldPolVecAfterCoordinates = distanceVectorCoordinatesInvariant(
                        Coordinates3D<double>(vec1.fX, vec1.fY, vec1.fZ),
                        Coordinates3D<double>(vec2.fX, vec2.fY, vec2.fZ), fieldDim);
                oldPolVecAfter.fX = oldPolVecAfterCoordinates.X();
                oldPolVecAfter.fY = oldPolVecAfterCoordinates.Y();
                oldPolVecAfter.fZ = oldPolVecAfterCoordinates.Z();
            }
        } else {
            oldPolVecAfter = Vector3(0.0, 0.0, 0.0);
        }
    }


    //calculate position of polarization vector for cluster of newCell after the flip
    if (newCell) {
        bool oldCellInvolved = false; //involved in polarization vector calculations
        if (newComp1 && newComp2) {
            Vector3 vec1; //corresponds to newComp1 after flip
            Vector3 vec2; //corresponds to newComp2 after flip
            if (newComp1 == newCell) {
                vec1 = newCOMAfter;
            } else if (newComp1 == oldCell) {
                vec1 = oldCOMAfter;
                oldCellInvolved = true;
            } else {
                vec1.fX = newComp1->xCOM;
                vec1.fY = newComp1->yCOM;
                vec1.fZ = newComp1->zCOM;
            }

            if (newComp2 == newCell) {
                vec2 = newCOMAfter;
            } else if (newComp2 == oldCell) {
                vec2 = oldCOMAfter;
                oldCellInvolved = true;
            } else {
                vec2.fX = newComp2->xCOM;
                vec2.fY = newComp2->yCOM;
                vec2.fZ = newComp2->zCOM;
            }

            if (oldCellAboutToDisappear && oldCellInvolved) {

//                 newPolVecAfter=Vector3(0.0,0.0,0.0);
                //if cell is about to disappear we do not impose any energy penalty in this plugin            
                newPolVecAfter = newPolVecBefore;
            } else {
                Coordinates3D<double> newPolVecAfterCoordinates = distanceVectorCoordinatesInvariant(
                        Coordinates3D<double>(vec1.fX, vec1.fY, vec1.fZ),
                        Coordinates3D<double>(vec2.fX, vec2.fY, vec2.fZ), fieldDim);
                newPolVecAfter.fX = newPolVecAfterCoordinates.X();
                newPolVecAfter.fY = newPolVecAfterCoordinates.Y();
                newPolVecAfter.fZ = newPolVecAfterCoordinates.Z();
            }
        } else {
            newPolVecAfter = Vector3(0.0, 0.0, 0.0);

        }

    }

    if (oldCell && newCell && oldCell->clusterId == newCell->clusterId) {
        //we will use before and after polarization vectors from oldCell -
        // this will automatically handle the situation of oldCell disappearing
        double eBefore = oldPolData.lambda * (oldPolData.polarizationVec - oldPolVecBefore) *
                         (oldPolData.polarizationVec - oldPolVecBefore);
        double eAfter = oldPolData.lambda * (oldPolData.polarizationVec - oldPolVecAfter) *
                        (oldPolData.polarizationVec - oldPolVecAfter);
        energy += (eAfter - eBefore);
    } else {
        if (newCell) {
            double eBefore = newPolData.lambda * (newPolData.polarizationVec - newPolVecBefore) *
                             (newPolData.polarizationVec - newPolVecBefore);
            double eAfter = newPolData.lambda * (newPolData.polarizationVec - newPolVecAfter) *
                            (newPolData.polarizationVec - newPolVecAfter);
            energy += (eAfter - eBefore);
        }
        if (oldCell) {
            double eBefore = oldPolData.lambda * (oldPolData.polarizationVec - oldPolVecBefore) *
                             (oldPolData.polarizationVec - oldPolVecBefore);
            double eAfter = oldPolData.lambda * (oldPolData.polarizationVec - oldPolVecAfter) *
                            (oldPolData.polarizationVec - oldPolVecAfter);
            energy += (eAfter - eBefore);

        }
    }


    return energy;
}


void Polarization23Plugin::setPolarizationVector(CellG *_cell, Vector3 &_vec) {
    if (!_cell)
        return;

    CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(_cell->clusterId);
    for (int i = 0; i < compartments.size(); ++i) {
        polarization23DataAccessor.get(compartments[i]->extraAttribPtr)->polarizationVec = _vec;
    }


}

Vector3 Polarization23Plugin::getPolarizationVector(CellG *_cell) {
    if (!_cell)
        return Vector3();
    return polarization23DataAccessor.get(_cell->extraAttribPtr)->polarizationVec;


}

void Polarization23Plugin::setPolarizationMarkers(CellG *_cell, unsigned char _type1, unsigned char _type2) {
    if (!_cell)
        return;

    CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(_cell->clusterId);
    for (int i = 0; i < compartments.size(); ++i) {
        polarization23DataAccessor.get(compartments[i]->extraAttribPtr)->type1 = _type1;
        polarization23DataAccessor.get(compartments[i]->extraAttribPtr)->type2 = _type2;
    }


}

std::vector<int> Polarization23Plugin::getPolarizationMarkers(CellG *_cell) {
    std::vector<int> typeVec(2, 0);
    if (!_cell) {
        return typeVec;
    }
    typeVec[0] = polarization23DataAccessor.get(_cell->extraAttribPtr)->type1;
    typeVec[1] = polarization23DataAccessor.get(_cell->extraAttribPtr)->type2;
    return typeVec;

}

void Polarization23Plugin::setLambdaPolarization(CellG *_cell, double _lambda) {
    if (!_cell)
        return;

    CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(_cell->clusterId);
    for (int i = 0; i < compartments.size(); ++i) {
        polarization23DataAccessor.get(compartments[i]->extraAttribPtr)->lambda = _lambda;
    }

}

double Polarization23Plugin::getLambdaPolarization(CellG *_cell) {
    if (!_cell) {
        return 0.0;
    }
    return polarization23DataAccessor.get(_cell->extraAttribPtr)->lambda;
}


void Polarization23Plugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();

}


std::string Polarization23Plugin::toString() {
    return "Polarization23";
}


std::string Polarization23Plugin::steerableName() {
    return toString();
}
