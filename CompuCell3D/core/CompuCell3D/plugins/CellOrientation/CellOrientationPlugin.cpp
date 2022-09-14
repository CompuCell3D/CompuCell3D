#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVector.h>
#include <CompuCell3D/plugins/PolarizationVector/PolarizationVectorPlugin.h>

using namespace CompuCell3D;

using namespace std;


#include "CellOrientationPlugin.h"


CellOrientationPlugin::CellOrientationPlugin() : potts(0),
                                                 simulator(0),
                                                 cellFieldG(0),
                                                 polarizationVectorAccessorPtr(0),
                                                 lambdaCellOrientation(0.0),
                                                 changeEnergyFcnPtr(&CellOrientationPlugin::changeEnergyPixelBased),
                                                 boundaryStrategy(0),
                                                 lambdaFlexFlag(false) {
}

void CellOrientationPlugin::setLambdaCellOrientation(CellG *_cell, double _lambda) {
    lambdaCellOrientationAccessor.get(_cell->extraAttribPtr)->lambdaVal = _lambda;
}

double CellOrientationPlugin::getLambdaCellOrientation(CellG *_cell) {
    return lambdaCellOrientationAccessor.get(_cell->extraAttribPtr)->lambdaVal;
}


CellOrientationPlugin::~CellOrientationPlugin() {

}

void CellOrientationPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
	CC3D_Log(LOG_DEBUG) << "INITIALIZE CELL ORIENTATION PLUGIN";
	potts = simulator->getPotts();
	//    potts->getCellFactoryGroupPtr()->registerClass(&CellOrientationVectorAccessor); //register new class with the factory

    bool pluginAlreadyRegisteredFlag;
    PolarizationVectorPlugin *polVectorPlugin = (PolarizationVectorPlugin *) Simulator::pluginManager.get(
            "PolarizationVector", &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        polVectorPlugin->init(simulator);

    bool comPluginAlreadyRegisteredFlag;

    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *comPlugin = Simulator::pluginManager.get("CenterOfMass",
                                                     &comPluginAlreadyRegisteredFlag);
    if (!comPluginAlreadyRegisteredFlag)
        comPlugin->init(simulator);

    polarizationVectorAccessorPtr = polVectorPlugin->getPolarizationVectorAccessorPtr();

    cellFieldG = potts->getCellFieldG();

    fieldDim = cellFieldG->getDim();

    boundaryStrategy = BoundaryStrategy::getInstance();

    potts->registerEnergyFunctionWithName(this, "CellOrientationEnergy");

    potts->getCellFactoryGroupPtr()->registerClass(&lambdaCellOrientationAccessor);


    simulator->registerSteerableObject(this);
    update(_xmlData, true);


}

void CellOrientationPlugin::extraInit(Simulator *simulator) {
    CC3D_Log(LOG_DEBUG) << "EXTRA INITIALIZE CELL ORIENTATION PLUGIN";
	Potts3D *potts = simulator->getPotts();
	cellFieldG = potts->getCellFieldG();
}


void CellOrientationPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    if (!_xmlData->getNumberOfChildren()) { //using cell id - based lambdaCellOrientation
        lambdaFlexFlag = true;
        return;
    }

    if (_xmlData->findElement("LambdaCellOrientation")) {
        lambdaCellOrientation = _xmlData->getFirstElement("LambdaCellOrientation")->getDouble();
    }

    if (_xmlData->findElement("LambdaFlex"))
        lambdaFlexFlag = true;
    else
        lambdaFlexFlag = false;

    bool comBasedAlgorithm = false;
    if (_xmlData->findElement("Algorithm")) {

        string algorithm = _xmlData->getFirstElement("Algorithm")->getText();

        changeToLower(algorithm);

        if (algorithm == "centerofmassbased") {
            comBasedAlgorithm = true;
            changeEnergyFcnPtr = &CellOrientationPlugin::changeEnergyCOMBased;
        }
    }

}

double CellOrientationPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    return 0.0;
}

double CellOrientationPlugin::changeEnergyPixelBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    float energy = 0.0;
    PolarizationVector *polarizationVecPtr;
    Point3D spinCopyVector;

    //this will return distance vector which will properly account for different boundary conditions
    spinCopyVector = distanceVectorInvariant(pt, potts->getFlipNeighbor(), fieldDim);


    double lambdaCellOrientationValue = 0.0;

    if (oldCell) {

        if (!lambdaFlexFlag) {
            lambdaCellOrientationValue = lambdaCellOrientation;
        } else {
            lambdaCellOrientationValue = lambdaCellOrientationAccessor.get(oldCell->extraAttribPtr)->lambdaVal;
        }

        polarizationVecPtr = polarizationVectorAccessorPtr->get(oldCell->extraAttribPtr);
        energy += -lambdaCellOrientationValue *
                  (polarizationVecPtr->x * spinCopyVector.x + polarizationVecPtr->y * spinCopyVector.y +
                   polarizationVecPtr->z * spinCopyVector.z);

    }


    if (newCell) {

        if (!lambdaFlexFlag) {
            lambdaCellOrientationValue = lambdaCellOrientation;
        } else {
            lambdaCellOrientationValue = lambdaCellOrientationAccessor.get(newCell->extraAttribPtr)->lambdaVal;
        }

        polarizationVecPtr = polarizationVectorAccessorPtr->get(newCell->extraAttribPtr);
        energy += -lambdaCellOrientationValue *
                  (polarizationVecPtr->x * spinCopyVector.x + polarizationVecPtr->y * spinCopyVector.y +
                   polarizationVecPtr->z * spinCopyVector.z);

    }

    return energy;
}


double CellOrientationPlugin::changeEnergyCOMBased(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    double energy = 0.0;
    PolarizationVector *polarizationVecPtr;
    double lambdaCellOrientationValue = 0.0;


    if (oldCell) {
        Coordinates3D<double> oldCOMAfterFlip = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);

        if (oldCell->volume > 1) {
            oldCOMAfterFlip.XRef() = oldCOMAfterFlip.X() / (float) (oldCell->volume - 1);
            oldCOMAfterFlip.YRef() = oldCOMAfterFlip.Y() / (float) (oldCell->volume - 1);
            oldCOMAfterFlip.ZRef() = oldCOMAfterFlip.Z() / (float) (oldCell->volume - 1);
        } else {

            oldCOMAfterFlip = Coordinates3D<double>(oldCell->xCM / oldCell->volume, oldCell->zCM / oldCell->volume,
                                                    oldCell->zCM / oldCell->volume);

        }

        if (!lambdaFlexFlag) {
            lambdaCellOrientationValue = lambdaCellOrientation;
        } else {
            lambdaCellOrientationValue = lambdaCellOrientationAccessor.get(oldCell->extraAttribPtr)->lambdaVal;
        }

        polarizationVecPtr = polarizationVectorAccessorPtr->get(oldCell->extraAttribPtr);


        Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                               oldCell->zCM / oldCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                              fieldDim);

        energy += -lambdaCellOrientationValue *
                  (polarizationVecPtr->x * distVector.x + polarizationVecPtr->y * distVector.y +
                   polarizationVecPtr->z * distVector.z);
    }


    if (newCell) {

        Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


        newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
        newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
        newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);

        if (!lambdaFlexFlag) {
            lambdaCellOrientationValue = lambdaCellOrientation;
        } else {
            lambdaCellOrientationValue = lambdaCellOrientationAccessor.get(newCell->extraAttribPtr)->lambdaVal;
        }

        polarizationVecPtr = polarizationVectorAccessorPtr->get(newCell->extraAttribPtr);

        Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                               newCell->zCM / newCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                              fieldDim);

        energy += -lambdaCellOrientationValue *
                  (polarizationVecPtr->x * distVector.x + polarizationVecPtr->y * distVector.y +
                   polarizationVecPtr->z * distVector.z);

    }

    return energy;
}


std::string CellOrientationPlugin::toString() {
    return "CellOrientation";
}


std::string CellOrientationPlugin::steerableName() {
    return toString();
}

