
/*
@author jfg
*/
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;


#include "ImplicitMotilityPlugin.h"

#include <math.h>
#include <Logger/CC3DLogger.h>


ImplicitMotilityPlugin::ImplicitMotilityPlugin() :

        pUtils(0),

        lockPtr(0),

        xmlData(0),

        cellFieldG(0),

        boundaryStrategy(0) {}


ImplicitMotilityPlugin::~ImplicitMotilityPlugin() {

    pUtils->destroyLock(lockPtr);

    delete lockPtr;

    lockPtr = 0;

}


void ImplicitMotilityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;

    sim = simulator;

    potts = simulator->getPotts();

    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();


    bool pluginAlreadyRegisteredFlag;
    //this will load CenterOfMass plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);


    pUtils = sim->getParallelUtils();

    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;

    pUtils->initLock(lockPtr);


    update(xmlData, true);


    potts->registerEnergyFunctionWithName(this, "ImplicitMotility");


    fieldDim = potts->getCellFieldG()->getDim();

    boundaryStrategy = BoundaryStrategy::getInstance();
    adjNeighbor.initialize(fieldDim);
    adjNeighbor_ptr = &adjNeighbor;

    if (potts->getBoundaryXName() == "Periodic") {
        adjNeighbor.setPeriodicX();
        boundaryConditionIndicator.x = 1;
    }
    if (potts->getBoundaryYName() == "Periodic") {
        adjNeighbor.setPeriodicY();
        boundaryConditionIndicator.y = 1;
    }
    if (potts->getBoundaryZName() == "Periodic") {
        adjNeighbor.setPeriodicZ();
        boundaryConditionIndicator.z = 1;
    }


    simulator->registerSteerableObject(this);

}


void ImplicitMotilityPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);

	bool steppableAlreadyRegisteredFlag;
    CC3D_Log(LOG_DEBUG) << "initializing the steppable";
    //this will load the bias vec steppable if it is not already
    Steppable *biasVectorSteppable = Simulator::steppableManager.get("BiasVectorSteppable",
                                                                     &steppableAlreadyRegisteredFlag);
    if (!steppableAlreadyRegisteredFlag) {
        biasVectorSteppable->init(simulator);
        ClassRegistry *class_registry = simulator->getClassRegistry();
        class_registry->addStepper("BiasVectorSteppable", biasVectorSteppable);

	}
    CC3D_Log(LOG_DEBUG) << "steppable initialized";

}


double ImplicitMotilityPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);
}


double ImplicitMotilityPlugin::changeEnergyByCellType(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {


    double energy = 0.0;
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

        Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                               oldCell->zCM / oldCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                              fieldDim);

        double norm = std::sqrt(
                distVector.X() * distVector.X() + distVector.Y() * distVector.Y() + distVector.Z() * distVector.Z());
        if (norm != 0) {
            distVector.XRef() = distVector.X() / norm;
            distVector.YRef() = distVector.Y() / norm;
            distVector.ZRef() = distVector.Z() / norm;
        }


        //Coordinates3D<double> biasVecTmp = oldCell->biasVector;
        biasVecTmp = Coordinates3D<double>(oldCell->biasVecX, oldCell->biasVecY, oldCell->biasVecZ);

        energy -= motilityParamMap[oldCell->type].lambdaMotility *
                  (distVector.X() * biasVecTmp.X() + distVector.Y() * biasVecTmp.Y() + distVector.Z() * biasVecTmp.Z());

    }

    if (newCell) {
        Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


        newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
        newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
        newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);


        Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                               newCell->zCM / newCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                              fieldDim);

        double norm = std::sqrt(
                distVector.X() * distVector.X() + distVector.Y() * distVector.Y() + distVector.Z() * distVector.Z());
        if (norm != 0) {
            distVector.XRef() = distVector.X() / norm;
            distVector.YRef() = distVector.Y() / norm;
            distVector.ZRef() = distVector.Z() / norm;
        }


        //Coordinates3D<double> biasVecTmp = newCell->biasVector;
        biasVecTmp = Coordinates3D<double>(newCell->biasVecX, newCell->biasVecY, newCell->biasVecZ);

        energy -= motilityParamMap[newCell->type].lambdaMotility *
                  (distVector.X() * biasVecTmp.X() + distVector.Y() * biasVecTmp.Y() + distVector.Z() * biasVecTmp.Z());
    }


    return energy;
}


double ImplicitMotilityPlugin::changeEnergyByCellId(const Point3D &pt, const CellG *newCell,
                                                    const CellG *oldCell) {


    double energy = 0.0;
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

        Coordinates3D<double> oldCOMBeforeFlip(oldCell->xCM / oldCell->volume, oldCell->yCM / oldCell->volume,
                                               oldCell->zCM / oldCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(oldCOMAfterFlip, oldCOMBeforeFlip,
                                                                              fieldDim);

        double norm = std::sqrt(
                distVector.X() * distVector.X() + distVector.Y() * distVector.Y() + distVector.Z() * distVector.Z());
        if (norm != 0) {
            distVector.XRef() = distVector.X() / norm;
            distVector.YRef() = distVector.Y() / norm;
            distVector.ZRef() = distVector.Z() / norm;
        }


        biasVecTmp = Coordinates3D<double>(oldCell->biasVecX, oldCell->biasVecY, oldCell->biasVecZ);

        energy -= oldCell->lambdaMotility *
                  (distVector.X() * biasVecTmp.X() + distVector.Y() * biasVecTmp.Y() + distVector.Z() * biasVecTmp.Z());
        //negative because it'd be confusing for users to have to define a negative lambda to go to a positive direction
    }


    if (newCell) {

        Coordinates3D<double> newCOMAfterFlip = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);


        newCOMAfterFlip.XRef() = newCOMAfterFlip.X() / (float) (newCell->volume + 1);
        newCOMAfterFlip.YRef() = newCOMAfterFlip.Y() / (float) (newCell->volume + 1);
        newCOMAfterFlip.ZRef() = newCOMAfterFlip.Z() / (float) (newCell->volume + 1);


        Coordinates3D<double> newCOMBeforeFlip(newCell->xCM / newCell->volume, newCell->yCM / newCell->volume,
                                               newCell->zCM / newCell->volume);
        Coordinates3D<double> distVector = distanceVectorCoordinatesInvariant(newCOMAfterFlip, newCOMBeforeFlip,
                                                                              fieldDim);

        double norm = std::sqrt(
                distVector.X() * distVector.X() + distVector.Y() * distVector.Y() + distVector.Z() * distVector.Z());
        if (norm != 0) {
            distVector.XRef() = distVector.X() / norm;
            distVector.YRef() = distVector.Y() / norm;
            distVector.ZRef() = distVector.Z() / norm;
        }

        //Coordinates3D<double> biasVecTmp = newCell->biasVector;
        biasVecTmp = Coordinates3D<double>(newCell->biasVecX, newCell->biasVecY, newCell->biasVecZ);


        energy -= newCell->lambdaMotility *
                  (distVector.X() * biasVecTmp.X() + distVector.Y() * biasVecTmp.Y() + distVector.Z() * biasVecTmp.Z());
        //negative because it'd be confusing for users to have to define a negative lambda to go to a positive direction
    }

    return energy;
}


void ImplicitMotilityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {


    //For more information on XML parser function please see CC3D code or lookup XML utils API

    automaton = potts->getAutomaton();

    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");

    set<unsigned char> cellTypesSet;

    boundaryStrategy = BoundaryStrategy::getInstance();


    if (_xmlData->findElement("MotilityEnergyParameters")) {
        functionType = BYCELLTYPE;
    } else {
        functionType = BYCELLID;
    }

    switch (functionType) {
        case BYCELLID:
            changeEnergyFcnPtr = &ImplicitMotilityPlugin::changeEnergyByCellId;
            break;
        case BYCELLTYPE: {
            motilityParamMap.clear();
            vector<unsigned char> typeIdVec;
            vector <ImplicitMotilityParam> motilityParamVectorTmp;

            CC3DXMLElementList energyVec = _xmlData->getElements("MotilityEnergyParameters");


            for (int i = 0; i < energyVec.size(); ++i) {
                ImplicitMotilityParam motParam;

                motParam.lambdaMotility = energyVec[i]->getAttributeAsDouble("LambdaMotility");
                motParam.typeName = energyVec[i]->getAttribute("CellType");
                typeIdVec.push_back(automaton->getTypeId(motParam.typeName));

                motilityParamVectorTmp.push_back(motParam);
            }

            vector<unsigned char>::iterator pos = max_element(typeIdVec.begin(), typeIdVec.end());


            int maxTypeId = 0;
            if (typeIdVec.size()) {
                maxTypeId = *pos;
            }

            for (int i = 0; i < motilityParamVectorTmp.size(); i++) {
                motilityParamMap[typeIdVec[i]] = motilityParamVectorTmp[i];
            }


            changeEnergyFcnPtr = &ImplicitMotilityPlugin::changeEnergyByCellType;
        }
            break;

        default:
            changeEnergyFcnPtr = &ImplicitMotilityPlugin::changeEnergyByCellId;

    }


}


std::string ImplicitMotilityPlugin::toString() {

    return "ImplicitMotility";

}


std::string ImplicitMotilityPlugin::steerableName() {

    return toString();

}

