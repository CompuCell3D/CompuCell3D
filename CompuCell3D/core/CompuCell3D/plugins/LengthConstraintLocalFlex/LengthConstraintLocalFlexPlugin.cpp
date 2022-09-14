#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "LengthConstraintLocalFlexPlugin.h"


LengthConstraintLocalFlexPlugin::LengthConstraintLocalFlexPlugin() : potts(0), boundaryStrategy(0),
                                                                     changeEnergyFcnPtr(0) {}

LengthConstraintLocalFlexPlugin::~LengthConstraintLocalFlexPlugin() {
}

void LengthConstraintLocalFlexPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    this->simulator = simulator;
    potts = simulator->getPotts();

    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("MomentOfInertia",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    boundaryStrategy = BoundaryStrategy::getInstance();

    potts->getCellFactoryGroupPtr()->registerClass(&lengthConstraintLocalFlexDataAccessor);
    potts->registerEnergyFunctionWithName(this, "LengthConstraintLocalFlex");


    Dim3D fieldDim = potts->getCellFieldG()->getDim();
    if (fieldDim.x == 1) {
        changeEnergyFcnPtr = &LengthConstraintLocalFlexPlugin::changeEnergy_yz;

    } else if (fieldDim.y == 1) {
        changeEnergyFcnPtr = &LengthConstraintLocalFlexPlugin::changeEnergy_xz;

    } else if (fieldDim.z == 1) {
        changeEnergyFcnPtr = &LengthConstraintLocalFlexPlugin::changeEnergy_xy;

    } else {
        ASSERT_OR_THROW("Currently LengthConstraint plugin can only be used in 2D", 0);
    }

}


void
LengthConstraintLocalFlexPlugin::setLengthConstraintData(CellG *_cell, double _lambdaLength, double _targetLength) {
    if (_cell) {
        lengthConstraintLocalFlexDataAccessor.get(_cell->extraAttribPtr)->lambdaLength = _lambdaLength;
        lengthConstraintLocalFlexDataAccessor.get(_cell->extraAttribPtr)->targetLength = _targetLength;
    }
}


double LengthConstraintLocalFlexPlugin::getTargetLength(CellG *_cell) {
    if (_cell) {
        return lengthConstraintLocalFlexDataAccessor.get(_cell->extraAttribPtr)->targetLength;
    }
    return 0.0;
}

double LengthConstraintLocalFlexPlugin::getLambdaLength(CellG *_cell) {
    if (_cell) {
        return lengthConstraintLocalFlexDataAccessor.get(_cell->extraAttribPtr)->lambdaLength;
    }
    return 0.0;

}


double LengthConstraintLocalFlexPlugin::changeEnergy(const Point3D &pt,
                                                     const CellG *newCell,
                                                     const CellG *oldCell) {

    /// E = lambda * (length - targetLength) ^ 2
    return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);

}

double LengthConstraintLocalFlexPlugin::changeEnergy_xz(const Point3D &pt,
                                                        const CellG *newCell,
                                                        const CellG *oldCell) {

    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done without checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving components of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    if (oldCell == newCell) return 0.0;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);

    //as in the original version
    if (newCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->targetLength;

        double xcm = (newCell->xCM / (float) newCell->volume);
        double zcm = (newCell->zCM / (float) newCell->volume);
        double newXCM = (newCell->xCM + ptTrans.x) / ((float) newCell->volume + 1);
        double newZCM = (newCell->zCM + ptTrans.z) / ((float) newCell->volume + 1);

        double newIxx = newCell->iXX + (newCell->volume) * zcm * zcm - (newCell->volume + 1) * (newZCM * newZCM) +
                        ptTrans.z * ptTrans.z;
        double newIzz = newCell->iZZ + (newCell->volume) * xcm * xcm - (newCell->volume + 1) * (newXCM * newXCM) +
                        ptTrans.x * ptTrans.x;
        double newIxz = newCell->iXZ - (newCell->volume) * xcm * zcm + (newCell->volume + 1) * newXCM * newZCM -
                        ptTrans.x * ptTrans.z;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (newCell->iXX + newCell->iZZ)) + .5 * sqrt((float) (
                (newCell->iXX - newCell->iZZ) * (newCell->iXX - newCell->iZZ) +
                4 * (newCell->iXZ) * (newCell->iXZ))))) / (float) (newCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((float) ((0.5 * (newIxx + newIzz)) + .5 * sqrt((float) (
                (newIxx - newIzz) * (newIxx - newIzz) + 4 * newIxz * newIxz)))) / (float) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->targetLength;

        double xcm = (oldCell->xCM / (float) oldCell->volume);
        double zcm = (oldCell->zCM / (float) oldCell->volume);
        double newXCM = (oldCell->xCM - ptTrans.x) / ((float) oldCell->volume - 1);
        double newZCM = (oldCell->zCM - ptTrans.z) / ((float) oldCell->volume - 1);

        double newIxx = oldCell->iXX + (oldCell->volume) * (zcm * zcm) - (oldCell->volume - 1) * (newZCM * newZCM) -
                        ptTrans.z * ptTrans.z;
        double newIzz = oldCell->iZZ + (oldCell->volume) * (xcm * xcm) - (oldCell->volume - 1) * (newXCM * newXCM) -
                        ptTrans.x * ptTrans.x;
        double newIxz = oldCell->iXZ - (oldCell->volume) * (xcm * zcm) + (oldCell->volume - 1) * newXCM * newZCM +
                        ptTrans.x * ptTrans.z;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (oldCell->iXX + oldCell->iZZ)) + .5 * sqrt((float) (
                (oldCell->iXX - oldCell->iZZ) * (oldCell->iXX - oldCell->iZZ) +
                4 * (oldCell->iXZ) * (oldCell->iXZ))))) / (float) (oldCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);

        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((float) ((0.5 * (newIxx + newIzz)) + .5 * sqrt((float) (
                    (newIxx - newIzz) * (newIxx - newIzz) + 4 * newIxz * newIxz)))) / (float) (oldCell->volume - 1));
        }


        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (energy != energy)
        return 0.0;
    else
        return energy;
}


double LengthConstraintLocalFlexPlugin::changeEnergy_xy(const Point3D &pt,
                                                        const CellG *newCell,
                                                        const CellG *oldCell) {

    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done without checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving components of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    if (oldCell == newCell) return 0.0;

    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);

    //as in the original version
    if (newCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->targetLength;

        double xcm = (newCell->xCM / (float) newCell->volume);
        double ycm = (newCell->yCM / (float) newCell->volume);
        double newXCM = (newCell->xCM + ptTrans.x) / ((float) newCell->volume + 1);
        double newYCM = (newCell->yCM + ptTrans.y) / ((float) newCell->volume + 1);

        double newIxx = newCell->iXX + (newCell->volume) * ycm * ycm - (newCell->volume + 1) * (newYCM * newYCM) +
                        ptTrans.y * ptTrans.y;
        double newIyy = newCell->iYY + (newCell->volume) * xcm * xcm - (newCell->volume + 1) * (newXCM * newXCM) +
                        ptTrans.x * ptTrans.x;
        double newIxy = newCell->iXY - (newCell->volume) * xcm * ycm + (newCell->volume + 1) * newXCM * newYCM -
                        ptTrans.x * ptTrans.y;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (newCell->iXX + newCell->iYY)) + .5 * sqrt((float) (
                (newCell->iXX - newCell->iYY) * (newCell->iXX - newCell->iYY) +
                4 * (newCell->iXY) * (newCell->iXY))))) / (float) (newCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((float) ((0.5 * (newIxx + newIyy)) + .5 * sqrt((float) (
                (newIxx - newIyy) * (newIxx - newIyy) + 4 * newIxy * newIxy)))) / (float) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->targetLength;

        double xcm = (oldCell->xCM / (float) oldCell->volume);
        double ycm = (oldCell->yCM / (float) oldCell->volume);
        double newXCM = (oldCell->xCM - ptTrans.x) / ((float) oldCell->volume - 1);
        double newYCM = (oldCell->yCM - ptTrans.y) / ((float) oldCell->volume - 1);

        double newIxx = oldCell->iXX + (oldCell->volume) * (ycm * ycm) - (oldCell->volume - 1) * (newYCM * newYCM) -
                        ptTrans.y * ptTrans.y;
        double newIyy = oldCell->iYY + (oldCell->volume) * (xcm * xcm) - (oldCell->volume - 1) * (newXCM * newXCM) -
                        ptTrans.x * ptTrans.x;
        double newIxy = oldCell->iXY - (oldCell->volume) * (xcm * ycm) + (oldCell->volume - 1) * newXCM * newYCM +
                        ptTrans.x * ptTrans.y;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (oldCell->iXX + oldCell->iYY)) + .5 * sqrt((float) (
                (oldCell->iXX - oldCell->iYY) * (oldCell->iXX - oldCell->iYY) +
                4 * (oldCell->iXY) * (oldCell->iXY))))) / (float) (oldCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((float) ((0.5 * (newIxx + newIyy)) + .5 * sqrt((float) (
                    (newIxx - newIyy) * (newIxx - newIyy) + 4 * newIxy * newIxy)))) / (float) (oldCell->volume - 1));
        }

        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (energy != energy)
        return 0.0;
    else
        return energy;
}

double LengthConstraintLocalFlexPlugin::changeEnergy_yz(const Point3D &pt,
                                                        const CellG *newCell,
                                                        const CellG *oldCell) {
    // Assumption: COM and Volume has not been updated.

    /// E = lambda * (length - targetLength) ^ 2

    //Center of mass, length constraints calculations are done without checking whether cell volume reaches 0 or not
    // when cell is about to disappear this results in Nan values of energy - because division by 0 is involved or
    // sqrt(expression involving components of inertia tensor) is NaN
    //in all such cases we set energy to 0 i.e. if energy=Nan we set it to energy=0.0

    double energy = 0.0;

    if (oldCell == newCell) return 0.0;
    Coordinates3D<double> ptTrans = boundaryStrategy->calculatePointCoordinates(pt);
    //as in the original version
    if (newCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(newCell->extraAttribPtr)->targetLength;

        double ycm = (newCell->yCM / (float) newCell->volume);
        double zcm = (newCell->zCM / (float) newCell->volume);
        double newYCM = (newCell->yCM + ptTrans.y) / ((float) newCell->volume + 1);
        double newZCM = (newCell->zCM + ptTrans.z) / ((float) newCell->volume + 1);

        double newIyy = newCell->iYY + (newCell->volume) * zcm * zcm - (newCell->volume + 1) * (newZCM * newZCM) +
                        ptTrans.z * ptTrans.z;
        double newIzz = newCell->iZZ + (newCell->volume) * ycm * ycm - (newCell->volume + 1) * (newYCM * newYCM) +
                        ptTrans.y * ptTrans.y;
        double newIyz = newCell->iYZ - (newCell->volume) * ycm * zcm + (newCell->volume + 1) * newYCM * newZCM -
                        ptTrans.y * ptTrans.z;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (newCell->iYY + newCell->iZZ)) + .5 * sqrt((float) (
                (newCell->iYY - newCell->iZZ) * (newCell->iYY - newCell->iZZ) +
                4 * (newCell->iYZ) * (newCell->iYZ))))) / (float) (newCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength = 4.0 * sqrt(((float) ((0.5 * (newIyy + newIzz)) + .5 * sqrt((float) (
                (newIyy - newIzz) * (newIyy - newIzz) + 4 * newIyz * newIyz)))) / (float) (newCell->volume + 1));
        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (oldCell) {
        double lambdaLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->lambdaLength;
        double targetLength = lengthConstraintLocalFlexDataAccessor.get(oldCell->extraAttribPtr)->targetLength;

        double ycm = (oldCell->yCM / (float) oldCell->volume);
        double zcm = (oldCell->zCM / (float) oldCell->volume);
        double newYCM = (oldCell->yCM - ptTrans.y) / ((float) oldCell->volume - 1);
        double newZCM = (oldCell->zCM - ptTrans.z) / ((float) oldCell->volume - 1);

        double newIyy = oldCell->iYY + (oldCell->volume) * (zcm * zcm) - (oldCell->volume - 1) * (newZCM * newZCM) -
                        ptTrans.z * ptTrans.z;
        double newIzz = oldCell->iZZ + (oldCell->volume) * (ycm * ycm) - (oldCell->volume - 1) * (newYCM * newYCM) -
                        ptTrans.y * ptTrans.y;
        double newIyz = oldCell->iYZ - (oldCell->volume) * (ycm * zcm) + (oldCell->volume - 1) * newYCM * newZCM +
                        ptTrans.y * ptTrans.z;

        double currLength = 4.0 * sqrt(((float) ((0.5 * (oldCell->iYY + oldCell->iZZ)) + .5 * sqrt((float) (
                (oldCell->iYY - oldCell->iZZ) * (oldCell->iYY - oldCell->iZZ) +
                4 * (oldCell->iYZ) * (oldCell->iYZ))))) / (float) (oldCell->volume));
        double currEnergy = lambdaLength * (currLength - targetLength) * (currLength - targetLength);
        double newLength;
        if (oldCell->volume <= 1) {
            newLength = 0.0;
        } else {
            newLength = 4.0 * sqrt(((float) ((0.5 * (newIyy + newIzz)) + .5 * sqrt((float) (
                    (newIyy - newIzz) * (newIyy - newIzz) + 4 * newIyz * newIyz)))) / (float) (oldCell->volume - 1));
        }

        double newEnergy = lambdaLength * (newLength - targetLength) * (newLength - targetLength);
        energy += newEnergy - currEnergy;
    }
    if (energy != energy)
        return 0.0;
    else
        return energy;
}

std::string LengthConstraintLocalFlexPlugin::toString() {
    return string("LengthConstraintLocalFlex");
}