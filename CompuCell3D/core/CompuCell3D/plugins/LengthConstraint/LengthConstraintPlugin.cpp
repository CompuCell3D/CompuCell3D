#include <CompuCell3D/CC3D.h>
#include <limits>

using namespace CompuCell3D;

using namespace std;


#include "LengthConstraintPlugin.h"

LengthConstraintPlugin::LengthConstraintPlugin() : xmlData(nullptr), potts(nullptr), changeEnergyFcnPtr(nullptr) {}

LengthConstraintPlugin::~LengthConstraintPlugin() = default;

void LengthConstraintPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;
    this->simulator = simulator;
    potts = simulator->getPotts();

    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("MomentOfInertia",
                                                  &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);


    boundaryStrategy = BoundaryStrategy::getInstance();

    potts->getCellFactoryGroupPtr()->registerClass(&lengthConstraintDataAccessor);
    potts->registerEnergyFunctionWithName(this, "LengthConstraint");


    simulator->registerSteerableObject(this);

    Dim3D fieldDim = potts->getCellFieldG()->getDim();
    if (fieldDim.x == 1) {
        changeEnergyFcnPtr = &LengthConstraintPlugin::changeEnergy_yz;

    } else if (fieldDim.y == 1) {
        changeEnergyFcnPtr = &LengthConstraintPlugin::changeEnergy_xz;

    } else if (fieldDim.z == 1) {
        changeEnergyFcnPtr = &LengthConstraintPlugin::changeEnergy_xy;

    } else {
        changeEnergyFcnPtr = &LengthConstraintPlugin::changeEnergy_3D;
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LengthConstraintPlugin::setLengthConstraintData(CellG *_cell, double _lambdaLength, double _targetLength,
                                                     double _minorTargetLength) {
    if (_cell) {
        lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->lambdaLength = _lambdaLength;
        lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->targetLength = _targetLength;
        lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->minorTargetLength = _minorTargetLength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getLambdaLength(CellG *_cell) {
    if (_cell) {
        return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->lambdaLength;
    }
    return 0.0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getTargetLength(CellG *_cell) {
    if (_cell) {
        return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->targetLength;
    }
    return 0.0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double LengthConstraintPlugin::getMinorTargetLength(CellG *_cell) {
    if (_cell) {
        return lengthConstraintDataAccessor.get(_cell->extraAttribPtr)->minorTargetLength;
    }
    return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LengthConstraintPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    typeNameVec.clear();
    lengthEnergyParamMap.clear();
    Automaton *automaton = potts->getAutomaton();

    CC3DXMLElementList lengthEnergyParamVecXML = _xmlData->getElements("LengthEnergyParameters");
    for (int i = 0; i < lengthEnergyParamVecXML.size(); ++i) {
        LengthEnergyParam lengthEnergyParam(
                lengthEnergyParamVecXML[i]->getAttribute("CellType"),
                lengthEnergyParamVecXML[i]->getAttributeAsDouble("TargetLength"),
                lengthEnergyParamVecXML[i]->getAttributeAsDouble("LambdaLength")
        );

        if (lengthEnergyParamVecXML[i]->findAttribute("MinorTargetLength")) {
            lengthEnergyParam.minorTargetLength = lengthEnergyParamVecXML[i]->getAttributeAsDouble("MinorTargetLength");
        }

        typeNameVec.push_back(lengthEnergyParam.cellTypeName);
        lengthEnergyParamMap[automaton->getTypeId(lengthEnergyParam.cellTypeName)] = lengthEnergyParam;
    }
    //have to make sure that potts ptr is initilized
    if (!potts) throw CC3DException("Potts pointer is unitialized");
}


void LengthConstraintPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);
}

double LengthConstraintPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {


    /// E = lambda * (length - targetLength) ^ 2
    if (oldCell == newCell) return 0.0;
    double energy = (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);

    return _get_non_nan_energy(energy);

}


double LengthConstraintPlugin::changeEnergy_xz(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {


    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving compoinents of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);
    //as in the original version
    if (newCell) {

        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(newCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }
        //we can optimize it further in case user does not specify local paramteress (i.e. per cell id and by-type definition is not specified as well)

        double xcm = (newCell->xCM / (double) newCell->volume);
        double zcm = (newCell->zCM / (double) newCell->volume);
        double newXCM = (newCell->xCM + ptTrans.x) / ((double) newCell->volume + 1);
        double newZCM = (newCell->zCM + ptTrans.z) / ((double) newCell->volume + 1);

        double newIxx = newCell->iXX + (newCell->volume) * zcm * zcm - (newCell->volume + 1) * (newZCM * newZCM) +
                        ptTrans.z * ptTrans.z;
        double newIzz = newCell->iZZ + (newCell->volume) * xcm * xcm - (newCell->volume + 1) * (newXCM * newXCM) +
                        ptTrans.x * ptTrans.x;
        double newIxz = newCell->iXZ - (newCell->volume) * xcm * zcm + (newCell->volume + 1) * newXCM * newZCM -
                        ptTrans.x * ptTrans.z;

        double currLength = 4.0 * sqrt(((double) ((0.5 * (newCell->iXX + newCell->iZZ)) + .5 * sqrt((double) (
                (newCell->iXX - newCell->iZZ) * (newCell->iXX - newCell->iZZ) +
                4 * (newCell->iXZ) * (newCell->iXZ))))) / (double) (newCell->volume));

        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((double) ((0.5 * (newIxx + newIzz)) + .5 * sqrt((double) (
                (newIxx - newIzz) * (newIxx - newIzz) + 4 * newIxz * newIxz)))) / (double) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(oldCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }

        double xcm = (oldCell->xCM / (double) oldCell->volume);
        double zcm = (oldCell->zCM / (double) oldCell->volume);
        double newXCM = (oldCell->xCM - ptTrans.x) / ((double) oldCell->volume - 1);
        double newZCM = (oldCell->zCM - ptTrans.z) / ((double) oldCell->volume - 1);

        double newIxx = oldCell->iXX + (oldCell->volume) * (zcm * zcm) - (oldCell->volume - 1) * (newZCM * newZCM) -
                        ptTrans.z * ptTrans.z;
        double newIzz = oldCell->iZZ + (oldCell->volume) * (xcm * xcm) - (oldCell->volume - 1) * (newXCM * newXCM) -
                        ptTrans.x * ptTrans.x;
        double newIxz = oldCell->iXZ - (oldCell->volume) * (xcm * zcm) + (oldCell->volume - 1) * newXCM * newZCM +
                        ptTrans.x * ptTrans.z;

        double currLength = 4.0 * sqrt(((double) ((0.5 * (oldCell->iXX + oldCell->iZZ)) + .5 * sqrt((double) (
                (oldCell->iXX - oldCell->iZZ) * (oldCell->iXX - oldCell->iZZ) +
                4 * (oldCell->iXZ) * (oldCell->iXZ))))) / (double) (oldCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((double) ((0.5 * (newIxx + newIzz)) + .5 * sqrt((double) (
                    (newIxx - newIzz) * (newIxx - newIzz) + 4 * newIxz * newIxz)))) / (double) (oldCell->volume - 1));
        }

        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    return energy;
}


double LengthConstraintPlugin::changeEnergy_xy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving compoinents of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);

    //as in the original version
    if (newCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(newCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }

        double xcm = (newCell->xCM / (double) newCell->volume);
        double ycm = (newCell->yCM / (double) newCell->volume);
        double newXCM = (newCell->xCM + ptTrans.x) / ((double) newCell->volume + 1);
        double newYCM = (newCell->yCM + ptTrans.y) / ((double) newCell->volume + 1);

        double newIxx = newCell->iXX + (newCell->volume) * ycm * ycm - (newCell->volume + 1) * (newYCM * newYCM) +
                        ptTrans.y * ptTrans.y;
        double newIyy = newCell->iYY + (newCell->volume) * xcm * xcm - (newCell->volume + 1) * (newXCM * newXCM) +
                        ptTrans.x * ptTrans.x;
        double newIxy = newCell->iXY - (newCell->volume) * xcm * ycm + (newCell->volume + 1) * newXCM * newYCM -
                        ptTrans.x * ptTrans.y;

        double currLength = 4.0 * sqrt(((double) ((0.5 * (newCell->iXX + newCell->iYY)) + .5 * sqrt((double) (
                (newCell->iXX - newCell->iYY) * (newCell->iXX - newCell->iYY) +
                4 * (newCell->iXY) * (newCell->iXY))))) / (double) (newCell->volume));

        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((double) ((0.5 * (newIxx + newIyy)) + .5 * sqrt((double) (
                (newIxx - newIyy) * (newIxx - newIyy) + 4 * newIxy * newIxy)))) / (double) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(oldCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }

        double xcm = (oldCell->xCM / (double) oldCell->volume);
        double ycm = (oldCell->yCM / (double) oldCell->volume);
        double newXCM = (oldCell->xCM - ptTrans.x) / ((double) oldCell->volume - 1);
        double newYCM = (oldCell->yCM - ptTrans.y) / ((double) oldCell->volume - 1);

        double newIxx = oldCell->iXX + (oldCell->volume) * (ycm * ycm) - (oldCell->volume - 1) * (newYCM * newYCM) -
                        ptTrans.y * ptTrans.y;
        double newIyy = oldCell->iYY + (oldCell->volume) * (xcm * xcm) - (oldCell->volume - 1) * (newXCM * newXCM) -
                        ptTrans.x * ptTrans.x;
        double newIxy = oldCell->iXY - (oldCell->volume) * (xcm * ycm) + (oldCell->volume - 1) * newXCM * newYCM +
                        ptTrans.x * ptTrans.y;

        double currLength = 4.0 * sqrt(((double) ((0.5 * (oldCell->iXX + oldCell->iYY)) + .5 * sqrt((double) (
                (oldCell->iXX - oldCell->iYY) * (oldCell->iXX - oldCell->iYY) +
                4 * (oldCell->iXY) * (oldCell->iXY))))) / (double) (oldCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);

        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((double) ((0.5 * (newIxx + newIyy)) + .5 * sqrt((double) (
                    (newIxx - newIyy) * (newIxx - newIyy) + 4 * newIxy * newIxy)))) / (double) (oldCell->volume - 1));
        }

        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);

        energy += newEnergy - currEnergy;
    }

    return energy;
}


double LengthConstraintPlugin::changeEnergy_yz(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving compoinents of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);
    //as in the original version
    if (newCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(newCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }

        double ycm = (newCell->yCM / (double) newCell->volume);
        double zcm = (newCell->zCM / (double) newCell->volume);
        double newYCM = (newCell->yCM + ptTrans.y) / ((double) newCell->volume + 1);
        double newZCM = (newCell->zCM + ptTrans.z) / ((double) newCell->volume + 1);

        double newIyy = newCell->iYY + (newCell->volume) * zcm * zcm - (newCell->volume + 1) * (newZCM * newZCM) +
                        ptTrans.z * ptTrans.z;
        double newIzz = newCell->iZZ + (newCell->volume) * ycm * ycm - (newCell->volume + 1) * (newYCM * newYCM) +
                        ptTrans.y * ptTrans.y;
        double newIyz = newCell->iYZ - (newCell->volume) * ycm * zcm + (newCell->volume + 1) * newYCM * newZCM -
                        ptTrans.y * ptTrans.z;


        double currLength = 4.0 * sqrt(((double) ((0.5 * (newCell->iYY + newCell->iZZ)) + .5 * sqrt((double) (
                (newCell->iYY - newCell->iZZ) * (newCell->iYY - newCell->iZZ) +
                4 * (newCell->iYZ) * (newCell->iYZ))))) / (double) (newCell->volume));

        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((double) ((0.5 * (newIyy + newIzz)) + .5 * sqrt((double) (
                (newIyy - newIzz) * (newIyy - newIzz) + 4 * newIyz * newIyz)))) / (double) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(oldCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
            }
        }

        double ycm = (oldCell->yCM / (double) oldCell->volume);
        double zcm = (oldCell->zCM / (double) oldCell->volume);
        double newYCM = (oldCell->yCM - ptTrans.y) / ((double) oldCell->volume - 1);
        double newZCM = (oldCell->zCM - ptTrans.z) / ((double) oldCell->volume - 1);

        double newIyy = oldCell->iYY + (oldCell->volume) * (zcm * zcm) - (oldCell->volume - 1) * (newZCM * newZCM) -
                        ptTrans.z * ptTrans.z;
        double newIzz = oldCell->iZZ + (oldCell->volume) * (ycm * ycm) - (oldCell->volume - 1) * (newYCM * newYCM) -
                        ptTrans.y * ptTrans.y;
        double newIyz = oldCell->iYZ - (oldCell->volume) * (ycm * zcm) + (oldCell->volume - 1) * newYCM * newZCM +
                        ptTrans.y * ptTrans.z;


        double currLength = 4.0 * sqrt(((double) ((0.5 * (oldCell->iYY + oldCell->iZZ)) + .5 * sqrt((double) (
                (oldCell->iYY - oldCell->iZZ) * (oldCell->iYY - oldCell->iZZ) +
                4 * (oldCell->iYZ) * (oldCell->iYZ))))) / (double) (oldCell->volume));

        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);

        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((double) ((0.5 * (newIyy + newIzz)) + .5 * sqrt((double) (
                    (newIyy - newIzz) * (newIyy - newIzz) + 4 * newIyz * newIyz)))) / (double) (oldCell->volume - 1));
        }


        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }

    return energy;

}

double LengthConstraintPlugin::spring_energy(double lam, double x, double x0) {
    return lam * pow(x - x0, 2.0);
}


double LengthConstraintPlugin::changeEnergy_3D(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done withou checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving compoinents of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0


    double energy = 0.0;
    double delta_en_new = 0.0;
    double delta_en = 0.0;

    double xcm, ycm, zcm, newXCM, newYCM, newZCM;

    double newIxx, newIyy, newIzz, newIxy, newIxz, newIyz;
    double lambdaLength, targetLength, minorTargetLength;

    double currEnergy = 0.0;
    double newLength = 0.0;
    double newMinorLength = 0.0;
    double newEnergy = 0.0;
    double currLength = 0.0;
    double currMinorLength = 0.0;


    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);

    //as in the original version
    if (newCell) {
        //local definitions of length constraint have priority over by type definitions
        double lambdaLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;;
        double targetLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->targetLength;;
        double minorTargetLength = lengthConstraintDataAccessor.get(newCell->extraAttribPtr)->minorTargetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(newCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
                minorTargetLength = lengthEnergyParamMapItr->second.minorTargetLength;
            }
        }

        xcm = (newCell->xCM / (float) newCell->volume);
        ycm = (newCell->yCM / (float) newCell->volume);
        zcm = (newCell->zCM / (float) newCell->volume);
        newXCM = (newCell->xCM + ptTrans.x) / ((float) newCell->volume + 1);
        newYCM = (newCell->yCM + ptTrans.y) / ((float) newCell->volume + 1);
        newZCM = (newCell->zCM + ptTrans.z) / ((float) newCell->volume + 1);

        newIxx = newCell->iXX + (newCell->volume) * (ycm * ycm + zcm * zcm) -
                 (newCell->volume + 1) * (newYCM * newYCM + newZCM * newZCM) + ptTrans.y * ptTrans.y +
                 ptTrans.z * ptTrans.z;
        newIyy = newCell->iYY + (newCell->volume) * (xcm * xcm + zcm * zcm) -
                 (newCell->volume + 1) * (newXCM * newXCM + newZCM * newZCM) + ptTrans.x * ptTrans.x +
                 ptTrans.z * ptTrans.z;
        newIzz = newCell->iZZ + (newCell->volume) * (xcm * xcm + ycm * ycm) -
                 (newCell->volume + 1) * (newXCM * newXCM + newYCM * newYCM) + ptTrans.x * ptTrans.x +
                 ptTrans.y * ptTrans.y;

        newIxy = newCell->iXY - (newCell->volume) * xcm * ycm + (newCell->volume + 1) * newXCM * newYCM -
                 ptTrans.x * ptTrans.y;
        newIxz = newCell->iXZ - (newCell->volume) * xcm * zcm + (newCell->volume + 1) * newXCM * newZCM -
                 ptTrans.x * ptTrans.z;
        newIyz = newCell->iYZ - (newCell->volume) * ycm * zcm + (newCell->volume + 1) * newYCM * newZCM -
                 ptTrans.y * ptTrans.z;

        vector<double> aCoeff(4, 0.0);
        vector<double> aCoeffNew(4, 0.0);
        vector <complex<double>> roots;
        vector <complex<double>> rootsNew;

        //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
        aCoeff[0] = -1.0;

        aCoeff[1] = newCell->iXX + newCell->iYY + newCell->iZZ;

        aCoeff[2] = newCell->iXY * newCell->iXY + newCell->iXZ * newCell->iXZ + newCell->iYZ * newCell->iYZ
                    - newCell->iXX * newCell->iYY - newCell->iXX * newCell->iZZ - newCell->iYY * newCell->iZZ;

        aCoeff[3] = newCell->iXX * newCell->iYY * newCell->iZZ + 2 * newCell->iXY * newCell->iXZ * newCell->iYZ
                    - newCell->iXX * newCell->iYZ * newCell->iYZ
                    - newCell->iYY * newCell->iXZ * newCell->iXZ
                    - newCell->iZZ * newCell->iXY * newCell->iXY;

        roots = solveCubicEquationRealCoeeficients(aCoeff);


        //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - after pixel copy

        aCoeffNew[0] = -1.0;

        aCoeffNew[1] = newIxx + newIyy + newIzz;

        aCoeffNew[2] = newIxy * newIxy + newIxz * newIxz + newIyz * newIyz
                       - newIxx * newIyy - newIxx * newIzz - newIyy * newIzz;

        aCoeffNew[3] = newIxx * newIyy * newIzz + 2 * newIxy * newIxz * newIyz
                       - newIxx * newIyz * newIyz
                       - newIyy * newIxz * newIxz
                       - newIzz * newIxy * newIxy;

        rootsNew = solveCubicEquationRealCoeeficients(aCoeffNew);


        //finding semiaxes of the ellipsoid
        //Ixx=m/5.0*(a_y^2+a_z^2) - andy cyclical permutations for other coordinate combinations
        //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
        // We can invert above system of equations to get:
        vector<double> axes(3, 0.0);

        axes[0] = sqrt((2.5 / newCell->volume) * (roots[1].real() + roots[2].real() - roots[0].real()));
        axes[1] = sqrt((2.5 / newCell->volume) * (roots[0].real() + roots[2].real() - roots[1].real()));
        axes[2] = sqrt((2.5 / newCell->volume) * (roots[0].real() + roots[1].real() - roots[2].real()));

        //sorting semiaxes according the their lengths (shortest first)
        sort(axes.begin(), axes.end());

        vector<double> axesNew(3, 0.0);

        axesNew[0] = sqrt(
                (2.5 / (newCell->volume + 1)) * (rootsNew[1].real() + rootsNew[2].real() - rootsNew[0].real()));
        axesNew[1] = sqrt(
                (2.5 / (newCell->volume + 1)) * (rootsNew[0].real() + rootsNew[2].real() - rootsNew[1].real()));
        axesNew[2] = sqrt(
                (2.5 / (newCell->volume + 1)) * (rootsNew[0].real() + rootsNew[1].real() - rootsNew[2].real()));

        //sorting semiaxes according the their lengths (shortest first)
        sort(axesNew.begin(), axesNew.end());

        currLength = 2.0 * axes[2];
        currMinorLength = 2.0 * axes[0];

        currEnergy = lambdaLength * ((currLength - targetLength) * (currLength - targetLength) +
                                     (currMinorLength - minorTargetLength) * (currMinorLength - minorTargetLength));

        newLength = 2.0 * axesNew[2];
        newMinorLength = 2.0 * axesNew[0];

        newEnergy = lambdaLength * ((newLength - targetLength) * (newLength - targetLength) +
                                    (newMinorLength - minorTargetLength) * (newMinorLength - minorTargetLength));

        energy += newEnergy - currEnergy;

    }


    if (oldCell) {
        
        lambdaLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;;
        targetLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->targetLength;;
        minorTargetLength = lengthConstraintDataAccessor.get(oldCell->extraAttribPtr)->minorTargetLength;;

        if (lambdaLength == 0.0) {
            auto lengthEnergyParamMapItr = lengthEnergyParamMap.find(oldCell->type);
            if (lengthEnergyParamMapItr != lengthEnergyParamMap.end()) {
                lambdaLength = lengthEnergyParamMapItr->second.lambdaLength;
                targetLength = lengthEnergyParamMapItr->second.targetLength;
                minorTargetLength = lengthEnergyParamMapItr->second.minorTargetLength;
            }
        }
        xcm = (oldCell->xCM / (float) oldCell->volume);
        ycm = (oldCell->yCM / (float) oldCell->volume);
        zcm = (oldCell->zCM / (float) oldCell->volume);
        newXCM = (oldCell->xCM - ptTrans.x) / ((float) oldCell->volume - 1);
        newYCM = (oldCell->yCM - ptTrans.y) / ((float) oldCell->volume - 1);
        newZCM = (oldCell->zCM - ptTrans.z) / ((float) oldCell->volume - 1);

        newIxx = oldCell->iXX + (oldCell->volume) * (ycm * ycm + zcm * zcm) -
                 (oldCell->volume - 1) * (newYCM * newYCM + newZCM * newZCM) -
                 (ptTrans.y * ptTrans.y + ptTrans.z * ptTrans.z);
        newIyy = oldCell->iYY + (oldCell->volume) * (xcm * xcm + zcm * zcm) -
                 (oldCell->volume - 1) * (newXCM * newXCM + newZCM * newZCM) -
                 (ptTrans.x * ptTrans.x + ptTrans.z * ptTrans.z);
        newIzz = oldCell->iZZ + (oldCell->volume) * (xcm * xcm + ycm * ycm) -
                 (oldCell->volume - 1) * (newXCM * newXCM + newYCM * newYCM) -
                 (ptTrans.x * ptTrans.x + ptTrans.y * ptTrans.y);

        newIxy = oldCell->iXY - (oldCell->volume) * (xcm * ycm) + (oldCell->volume - 1) * newXCM * newYCM +
                 ptTrans.x * ptTrans.y;
        newIxz = oldCell->iXZ - (oldCell->volume) * (xcm * zcm) + (oldCell->volume - 1) * newXCM * newZCM +
                 ptTrans.x * ptTrans.z;
        newIyz = oldCell->iYZ - (oldCell->volume) * (ycm * zcm) + (oldCell->volume - 1) * newYCM * newZCM +
                 ptTrans.y * ptTrans.z;


        vector<double> aCoeff(4, 0.0);
        vector<double> aCoeffNew(4, 0.0);
        vector <complex<double>> roots;
        vector <complex<double>> rootsNew;

        //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - before pixel copy
        aCoeff[0] = -1.0;

        aCoeff[1] = oldCell->iXX + oldCell->iYY + oldCell->iZZ;

        aCoeff[2] = oldCell->iXY * oldCell->iXY + oldCell->iXZ * oldCell->iXZ + oldCell->iYZ * oldCell->iYZ
                    - oldCell->iXX * oldCell->iYY - oldCell->iXX * oldCell->iZZ - oldCell->iYY * oldCell->iZZ;

        aCoeff[3] = oldCell->iXX * oldCell->iYY * oldCell->iZZ + 2 * oldCell->iXY * oldCell->iXZ * oldCell->iYZ
                    - oldCell->iXX * oldCell->iYZ * oldCell->iYZ
                    - oldCell->iYY * oldCell->iXZ * oldCell->iXZ
                    - oldCell->iZZ * oldCell->iXY * oldCell->iXY;

        roots = solveCubicEquationRealCoeeficients(aCoeff);


        //initialize coefficients of cubic eq used to find eigenvalues of inertia tensor - after pixel copy

        aCoeffNew[0] = -1.0;

        aCoeffNew[1] = newIxx + newIyy + newIzz;

        aCoeffNew[2] = newIxy * newIxy + newIxz * newIxz + newIyz * newIyz
                       - newIxx * newIyy - newIxx * newIzz - newIyy * newIzz;

        aCoeffNew[3] = newIxx * newIyy * newIzz + 2 * newIxy * newIxz * newIyz
                       - newIxx * newIyz * newIyz
                       - newIyy * newIxz * newIxz
                       - newIzz * newIxy * newIxy;

        rootsNew = solveCubicEquationRealCoeeficients(aCoeffNew);

        //finding semiaxes of the ellipsoid
        //Ixx=m/5.0*(a_y^2+a_z^2) - and cyclical permutations for other coordinate combinations
        //a_x,a_y,a_z are lengths of semiaxes of the allipsoid
        // We can invert above system of equations to get:
        vector<double> axes(3, 0.0);

        axes[0] = sqrt((2.5 / oldCell->volume) * (roots[1].real() + roots[2].real() - roots[0].real()));
        axes[1] = sqrt((2.5 / oldCell->volume) * (roots[0].real() + roots[2].real() - roots[1].real()));
        axes[2] = sqrt((2.5 / oldCell->volume) * (roots[0].real() + roots[1].real() - roots[2].real()));

        //sorting semiaxes according the their lengths (shortest first)
        sort(axes.begin(), axes.end());

        vector<double> axesNew(3, 0.0);
        if (oldCell->volume <= 1) {
            axesNew[0] = 0.0;
            axesNew[1] = 0.0;
            axesNew[2] = 0.0;
        } else {
            axesNew[0] = sqrt(
                    (2.5 / (oldCell->volume - 1)) * (rootsNew[1].real() + rootsNew[2].real() - rootsNew[0].real()));
            axesNew[1] = sqrt(
                    (2.5 / (oldCell->volume - 1)) * (rootsNew[0].real() + rootsNew[2].real() - rootsNew[1].real()));
            axesNew[2] = sqrt(
                    (2.5 / (oldCell->volume - 1)) * (rootsNew[0].real() + rootsNew[1].real() - rootsNew[2].real()));
        }
        //sorting semiaxes according the their lengths (shortest first)
        sort(axesNew.begin(), axesNew.end());

        currLength = 2.0 * axes[2];
        currMinorLength = 2.0 * axes[0];

        currEnergy = lambdaLength * ((currLength - targetLength) * (currLength - targetLength) +
                                     (currMinorLength - minorTargetLength) * (currMinorLength - minorTargetLength));

        newLength = 2.0 * axesNew[2];
        newMinorLength = 2.0 * axesNew[0];

        newEnergy = lambdaLength * ((newLength - targetLength) * (newLength - targetLength) +
                                    (newMinorLength - minorTargetLength) * (newMinorLength - minorTargetLength));

        energy += newEnergy - currEnergy;

    }

    return energy;

}

double LengthConstraintPlugin::_get_non_nan_energy(double energy) {
    if (energy != energy)
        return 0.0;
    else
        return energy;
}


std::string LengthConstraintPlugin::toString() {
    return string("LengthConstraint");
}

std::string LengthConstraintPlugin::steerableName() {

    return toString();

}

