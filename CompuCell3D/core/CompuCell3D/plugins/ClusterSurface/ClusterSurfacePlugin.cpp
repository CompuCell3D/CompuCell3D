#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/plugins/ClusterSurfaceTracker/ClusterSurfaceTrackerPlugin.h>

#include "ClusterSurfacePlugin.h"


ClusterSurfacePlugin::ClusterSurfacePlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        scaleClusterSurface(1.0),
        boundaryStrategy(0),
        changeEnergyFcnPtr(&ClusterSurfacePlugin::changeEnergyByCellId) {}

ClusterSurfacePlugin::~ClusterSurfacePlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void ClusterSurfacePlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);


    //This code is usually called from   init function
    bool pluginAlreadyRegisteredFlag;
    //this will load PLUGIN_NAME plugin if it is not already loaded
    cstPlugin = (ClusterSurfaceTrackerPlugin *) Simulator::pluginManager.get("ClusterSurfaceTracker",
                                                                             &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        cstPlugin->init(simulator);

    update(xmlData, true);

    maxNeighborIndex = cstPlugin->getMaxNeighborIndex();
    lmf = cstPlugin->getLatticeMultiplicativeFactors();

    potts->registerEnergyFunctionWithName(this, "ClusterSurface");

    simulator->registerSteerableObject(this);
}
// ExtraInit functions are called in the order plugin was fisrt loaded to set of active plugins inside pluginManager
// Notice that this order is in general not the same as the order of two trackers then tracker 2 inserts tracker
// 1 before itself in the list of trackers
// If tracker 2 is first accessed before tracker 1 then order pluginManager order is tracker 2,
// tracker 1 whereas tracker registry has them ordered tracker 1 , tracker 2
// the bottom line is that we can rely on ordering in tracker registry, energy registry etc.
// but cannot rely on ordering when it comes to extraInit fcns
// All the order dependent action should be preferable be done in the init fcns.
// and of course init is always called before extra init so this gives extra wiggle room

void ClusterSurfacePlugin::extraInit(Simulator *simulator) {

}

void ClusterSurfacePlugin::setTargetAndLambdaClusterSurface(CellG *_cell, float _targetClusterSurface,
                                                            float _lambdaClusterSurface) {

    CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(_cell->clusterId);

    for (int i = 0; i < compartments.size(); ++i) {
        compartments[i]->targetClusterSurface = _targetClusterSurface;
        compartments[i]->lambdaClusterSurface = _lambdaClusterSurface;
    }

}

pair<float, float> ClusterSurfacePlugin::getTargetAndLambdaVolume(const CellG *_cell) const {
    return make_pair(_cell->targetClusterSurface, _cell->lambdaClusterSurface);
}


std::pair<double, double>
ClusterSurfacePlugin::getNewOldClusterSurfaceDiffs(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    CellG *nCell;
    double oldDiff = 0.;
    double newDiff = 0.;

    Neighbor neighbor;
    if (oldCell == newCell) return make_pair(0.0, 0.0);

    if (oldCell && newCell && oldCell->clusterId == newCell->clusterId) return make_pair(0.0, 0.0);

    //we calculate gain/loss of surface for clusters of oldCell and newCell -
    // we assume here that new and old cell belong to different clusters

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);

        if (!neighbor.distance)
            continue;

        nCell = cellFieldG->get(neighbor.pt);


        if (newCell && nCell && newCell->clusterId == nCell->clusterId) {
            newDiff -= lmf.surfaceMF;
        } else {
            newDiff += lmf.surfaceMF;
        }


        if (oldCell && nCell && oldCell->clusterId == nCell->clusterId) {
            oldDiff += lmf.surfaceMF;
        } else {
            oldDiff -= lmf.surfaceMF;
        }

    }

    return make_pair(newDiff, oldDiff);
}

double
ClusterSurfacePlugin::diffEnergy(double lambda, double targetClusterSurface, double clusterSurface, double diff) {
    return lambda * (diff * diff + 2 * diff * (clusterSurface - fabs(targetClusterSurface)));
}

double ClusterSurfacePlugin::changeEnergyByCellId(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    /// E = lambda * (ClusterSurface - targetClusterSurface) ^ 2
    double energy = 0.0;

    if (oldCell == newCell) return 0.0;
    if (oldCell && newCell && oldCell->clusterId == newCell->clusterId) return 0.0;

    pair<double, double> newOldDiffs = getNewOldClusterSurfaceDiffs(pt, newCell, oldCell);

    if (newCell) {
        energy += diffEnergy(newCell->lambdaClusterSurface, newCell->targetClusterSurface,
                             newCell->clusterSurface * scaleClusterSurface, newOldDiffs.first * scaleClusterSurface);

    }
    if (oldCell) {
        energy += diffEnergy(oldCell->lambdaClusterSurface, oldCell->targetClusterSurface,
                             oldCell->clusterSurface * scaleClusterSurface, newOldDiffs.second * scaleClusterSurface);
    }

    return energy;

}

double ClusterSurfacePlugin::changeEnergyGlobal(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
    /// E = lambda * (ClusterSurface - targetClusterSurface) ^ 2 
    double energy = 0.0;

    if (oldCell == newCell) return 0.0;
    if (oldCell && newCell && oldCell->clusterId == newCell->clusterId) return 0.0;

    pair<double, double> newOldDiffs = getNewOldClusterSurfaceDiffs(pt, newCell, oldCell);

    if (newCell) {
        energy += diffEnergy(lambdaClusterSurface, targetClusterSurface, newCell->clusterSurface * scaleClusterSurface,
                             newOldDiffs.first * scaleClusterSurface);

    }
    if (oldCell) {
        energy += diffEnergy(lambdaClusterSurface, targetClusterSurface, oldCell->clusterSurface * scaleClusterSurface,
                             newOldDiffs.second * scaleClusterSurface);
    }


    return energy;

}


double ClusterSurfacePlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    /// E = lambda * (ClusterSurface - targetClusterSurface) ^ 2 
    return (this->*changeEnergyFcnPtr)(pt, newCell, oldCell);

}


void ClusterSurfacePlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    //if(potts->getDisplayUnitsFlag()){
    //    Unit targetSurfaceUnit=powerUnit(potts->getLengthUnit(),2);
    //    Unit lambdaSurfaceUnit=potts->getEnergyUnit()/(targetSurfaceUnit*targetSurfaceUnit);

    //    CC3DXMLElement * unitsElem=_xmlData->getFirstElement("Units"); 
    //    if (!unitsElem){ //add Units element
    //        unitsElem=_xmlData->attachElement("Units");
    //    }

    //    if(unitsElem->getFirstElement("TargetSurfaceUnit")){
    //        unitsElem->getFirstElement("TargetSurfaceUnit")->updateElementValue(targetSurfaceUnit.toString());
    //    }else{
    //        CC3DXMLElement * surfaceUnitElem = unitsElem->attachElement("TargetSurfaceUnit",targetSurfaceUnit.toString());
    //    }

    //    if(unitsElem->getFirstElement("LambdaSurfaceUnit")){
    //        unitsElem->getFirstElement("LambdaSurfaceUnit")->updateElementValue(lambdaSurfaceUnit.toString());
    //    }else{
    //        CC3DXMLElement * lambdaSurfaceUnitElem = unitsElem->attachElement("LambdaSurfaceUnit",lambdaSurfaceUnit.toString());
    //    }


    //}


    //if there are no child elements for this plugin it means will use changeEnergyByCellId
    if (!_xmlData->getNumberOfChildren()) {
        functionType = BYCELLID;
    } else {
        if (_xmlData->findElement("TargetClusterSurface"))
            functionType = GLOBAL;
        else //in case users put garbage xml use changeEnergyByCellId
            functionType = BYCELLID;
    }
    Automaton *automaton = potts->getAutomaton();
    CC3D_Log(LOG_DEBUG) << "automaton="<<automaton;

    switch (functionType) {
        case BYCELLID:
            //set fcn ptr
            changeEnergyFcnPtr = &ClusterSurfacePlugin::changeEnergyByCellId;
            break;

        case GLOBAL:
            //using Global Surface Energy Parameters
            targetClusterSurface = _xmlData->getFirstElement("TargetClusterSurface")->getDouble();
            lambdaClusterSurface = _xmlData->getFirstElement("LambdaClusterSurface")->getDouble();

            //set fcn ptr
            changeEnergyFcnPtr = &ClusterSurfacePlugin::changeEnergyGlobal;
            break;

        default:
            //set fcn ptr
            changeEnergyFcnPtr = &ClusterSurfacePlugin::changeEnergyByCellId;
    }
    //check if there is a ScaleSurface parameter  in XML
    if (_xmlData->findElement("ScaleClusterSurface")) {
        scaleClusterSurface = _xmlData->getFirstElement("ScaleClusterSurface")->getDouble();
    }

    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();

}


std::string ClusterSurfacePlugin::toString() {
    return "ClusterSurface";
}


std::string ClusterSurfacePlugin::steerableName() {
    return toString();
}
