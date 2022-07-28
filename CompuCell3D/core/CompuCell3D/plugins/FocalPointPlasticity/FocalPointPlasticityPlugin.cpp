#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;

#include "FocalPointPlasticityPlugin.h"


FocalPointPlasticityPlugin::FocalPointPlasticityPlugin() : pUtils(0), xmlData(0) {
    lambda = 0.0;
    activationEnergy = 0.0;

    targetDistance = 0.0;


    maxDistance = 1000.0;

    diffEnergyFcnPtr = &FocalPointPlasticityPlugin::diffEnergyByType;

    //setting default elastic link constituent law
    constituentLawFcnPtr = &FocalPointPlasticityPlugin::elasticLinkConstituentLaw;

    functionType = BYCELLTYPE;
    neighborOrder = 1;
}

FocalPointPlasticityPlugin::~FocalPointPlasticityPlugin() {

}

void FocalPointPlasticityPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    sim = simulator;
    potts = simulator->getPotts();
    fieldDim = potts->getCellFieldG()->getDim();
    xmlData = _xmlData;
    simulator->getPotts()->registerEnergyFunctionWithName(this, toString());
    simulator->registerSteerableObject(this);

    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                  &pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);

    //first need to register center of mass plugin and then register FocalPointPlasticity
    potts->registerCellGChangeWatcher(this);
    // Instantiate these here for safe garbage collection
    potts->getCellFactoryGroupPtr()->registerClass(&cellLinkInventoryTracker);
    potts->getCellFactoryGroupPtr()->registerClass(&cellInternalLinkInventoryTracker);
    potts->getCellFactoryGroupPtr()->registerClass(&cellAnchorInventoryTracker);
    // Initialize link inventories
    linkInv = FPPLinkInventory(&cellLinkInventoryTracker, potts);
    linkInvInternal = FPPInternalLinkInventory(&cellInternalLinkInventoryTracker, potts);
    linkInvAnchor = FPPAnchorInventory(&cellAnchorInventoryTracker, potts);

    pUtils = simulator->getParallelUtils();
    unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
    newJunctionInitiatedFlagVec.assign(maxNumberOfWorkNodes, false);
    newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes, false);
    newNeighborVec.assign(maxNumberOfWorkNodes, 0);

}

void FocalPointPlasticityPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);
}

void FocalPointPlasticityPlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id == CHANGE_NUMBER_OF_WORK_NODES) {
        unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
        newJunctionInitiatedFlagVec.assign(maxNumberOfWorkNodes, false);
        newJunctionInitiatedFlagWithinClusterVec.assign(maxNumberOfWorkNodes, false);
        newNeighborVec.assign(maxNumberOfWorkNodes, 0);

        update(xmlData);
    }

}


void FocalPointPlasticityPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();

    plastParamsArray.clear();
    internalPlastParamsArray.clear();

    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");

    CC3DXMLElementList plastParamVec = _xmlData->getElements("Parameters");
    if (plastParamVec.size() > 0) {
        functionType = BYCELLTYPE;
    }

    if (_xmlData->getFirstElement("Local")) {
        diffEnergyFcnPtr = &FocalPointPlasticityPlugin::diffEnergyLocal;
        functionType = BYCELLID;
    }

    if (_xmlData->getFirstElement("NeighborOrder")) {
        neighborOrder = _xmlData->getFirstElement("NeighborOrder")->getInt();
    }

    for (int i = 0; i < plastParamVec.size(); ++i) {

        FocalPointPlasticityTrackerData fpptd;

        unsigned char type1 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type1"));
        unsigned char type2 = automaton->getTypeId(plastParamVec[i]->getAttribute("Type2"));

        if (plastParamVec[i]->getFirstElement("Lambda"))
            fpptd.lambdaDistance = plastParamVec[i]->getFirstElement("Lambda")->getDouble();

        if (plastParamVec[i]->getFirstElement("TargetDistance"))
            fpptd.targetDistance = plastParamVec[i]->getFirstElement("TargetDistance")->getDouble();

        if (plastParamVec[i]->getFirstElement("ActivationEnergy")) {
            fpptd.activationEnergy = plastParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();
        }

        if (plastParamVec[i]->getFirstElement("MaxDistance"))
            fpptd.maxDistance = plastParamVec[i]->getFirstElement("MaxDistance")->getDouble();

        if (plastParamVec[i]->getFirstElement("MaxNumberOfJunctions")) {
            CC3DXMLElement *maxNumberOfJunctionsElement = plastParamVec[i]->getFirstElement("MaxNumberOfJunctions");
            fpptd.maxNumberOfJunctions = maxNumberOfJunctionsElement->getInt();
        }

        plastParamsArray[type1][type2] = fpptd;
        plastParamsArray[type2][type1] = fpptd;
    }


    // extracting internal parameters - used with compartmental cells
    CC3DXMLElementList internalPlastParamVec = _xmlData->getElements("InternalParameters");
    for (int i = 0; i < internalPlastParamVec.size(); ++i) {

        FocalPointPlasticityTrackerData fpptd;

        unsigned char type1 = automaton->getTypeId(internalPlastParamVec[i]->getAttribute("Type1"));
        unsigned char type2 = automaton->getTypeId(internalPlastParamVec[i]->getAttribute("Type2"));

        if (internalPlastParamVec[i]->getFirstElement("Lambda"))
            fpptd.lambdaDistance = internalPlastParamVec[i]->getFirstElement("Lambda")->getDouble();

        if (internalPlastParamVec[i]->getFirstElement("TargetDistance"))
            fpptd.targetDistance = internalPlastParamVec[i]->getFirstElement("TargetDistance")->getDouble();

        if (internalPlastParamVec[i]->getFirstElement("ActivationEnergy"))
            fpptd.activationEnergy = internalPlastParamVec[i]->getFirstElement("ActivationEnergy")->getDouble();

        if (internalPlastParamVec[i]->getFirstElement("MaxDistance"))
            fpptd.maxDistance = internalPlastParamVec[i]->getFirstElement("MaxDistance")->getDouble();

        if (internalPlastParamVec[i]->getFirstElement("MaxNumberOfJunctions")) {
            CC3DXMLElement *maxNumberOfJunctionsElement = internalPlastParamVec[i]->getFirstElement(
                    "MaxNumberOfJunctions");
            fpptd.maxNumberOfJunctions = maxNumberOfJunctionsElement->getInt();

        }

        internalPlastParamsArray[type1][type2] = fpptd;
        internalPlastParamsArray[type2][type1] = fpptd;
    }

    maxNumberOfJunctionsTotalMap.clear();
    for (auto &itr: plastParamsArray) {
        int mNJ = 0;
        for (auto &itrItr: itr.second) {

            mNJ += itrItr.second.maxNumberOfJunctions;
        }
        maxNumberOfJunctionsTotalMap[itr.first] = mNJ;
    }

    //Now internal parameters

    //initializing maxNumberOfJunctionsInternalTotalVec based on plastParamsArray .
    // maxNumberOfJunctionsInternalTotalVec is indexed by cell type
    maxNumberOfJunctionsInternalTotalMap.clear();
    for (auto &itr: internalPlastParamsArray) {
        int mNJ = 0;
        for (auto &itrItr: itr.second) {

            mNJ += itrItr.second.maxNumberOfJunctions;
        }
        maxNumberOfJunctionsInternalTotalMap[itr.first] = mNJ;
    }

    CC3DXMLElement *linkXMLElem = _xmlData->getFirstElement("LinkConstituentLaw");

    if (linkXMLElem && linkXMLElem->findElement("Formula")) {
        throw CC3DException(
                "CC3DML Error: Please change Formula tag to Expression tag inside LinkConstituentLaw element");
    }

    if (linkXMLElem) {
        unsigned int maxNumberOfWorkNodes = pUtils->getMaxNumberOfWorkNodesPotts();
        eed.allocateSize(maxNumberOfWorkNodes);
        vector <string> variableNames;
        variableNames.push_back("Lambda");
        variableNames.push_back("Length");
        variableNames.push_back("TargetLength");


        eed.addVariables(variableNames.begin(), variableNames.end());
        eed.update(linkXMLElem);
        constituentLawFcnPtr = &FocalPointPlasticityPlugin::customLinkConstituentLaw;

    } else {

        constituentLawFcnPtr = &FocalPointPlasticityPlugin::elasticLinkConstituentLaw;;
    }

    boundaryStrategy = BoundaryStrategy::getInstance();

}

double FocalPointPlasticityPlugin::potentialFunction(double _lambda, double _offset, double _targetDistance,
                                                     double _distance) {
    return _offset + _lambda * (_distance - _targetDistance) * (_distance - _targetDistance);
}

double FocalPointPlasticityPlugin::elasticLinkConstituentLaw(float _lambda, float _length, float _targetLength) {

    return _lambda * (_length - _targetLength) * (_length - _targetLength);

}

double FocalPointPlasticityPlugin::customLinkConstituentLaw(float _lambda, float _length, float _targetLength) {

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    ExpressionEvaluator &ev = eed[currentWorkNodeNumber];
    double linkLaw = 0.0;

    ev[0] = _lambda;
    ev[1] = _length;
    ev[2] = _targetLength;

    linkLaw = ev.eval();

    return linkLaw;

}

double FocalPointPlasticityPlugin::diffEnergyLocal(float _deltaL, float _lBefore,
                                                   const FocalPointPlasticityTrackerData &_plasticityTrackerData,
                                                   const CellG *_cell, bool _useCluster) {

    float lambdaLocal = _plasticityTrackerData.lambdaDistance;
    float targetDistanceLocal = _plasticityTrackerData.targetDistance;

    if (_cell->volume > 1) {

        return (this->*constituentLawFcnPtr)(lambdaLocal, _lBefore + _deltaL, targetDistanceLocal) -
               (this->*constituentLawFcnPtr)(lambdaLocal, _lBefore, targetDistanceLocal);
    } else {//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
        return -(this->*constituentLawFcnPtr)(lambdaLocal, _lBefore, targetDistanceLocal);
    }

}

double FocalPointPlasticityPlugin::diffEnergyByType(float _deltaL, float _lBefore,
                                                    const FocalPointPlasticityTrackerData &_plasticityTrackerData,
                                                    const CellG *_cell, bool _useCluster) {
    float lambdaDistanceLocal;
    float targetDistanceLocal;
    if (_useCluster) {
        lambdaDistanceLocal = internalPlastParamsArray[_plasticityTrackerData.neighborAddress->type][_cell->type].lambdaDistance;
        targetDistanceLocal = internalPlastParamsArray[_plasticityTrackerData.neighborAddress->type][_cell->type].targetDistance;
    } else {
        lambdaDistanceLocal = plastParamsArray[_plasticityTrackerData.neighborAddress->type][_cell->type].lambdaDistance;
        targetDistanceLocal = plastParamsArray[_plasticityTrackerData.neighborAddress->type][_cell->type].targetDistance;
    }

    if (_cell->volume > 1) {
        return (this->*constituentLawFcnPtr)(lambdaDistanceLocal, _lBefore + _deltaL, targetDistanceLocal) -
               (this->*constituentLawFcnPtr)(lambdaDistanceLocal, _lBefore, targetDistanceLocal);
    } else {//after spin flip oldCell will disappear so the only contribution from before spin flip i.e. -(l-l0)^2
        return -(this->*constituentLawFcnPtr)(lambdaDistanceLocal, _lBefore, targetDistanceLocal);
    }
}

void FocalPointPlasticityPlugin::insertFPPData(CellG *_cell, FocalPointPlasticityTrackerData *_fpptd) {

    CellG *_cell1, *_cell2;
    if (_fpptd->isInitiator) {
        _cell1 = _cell;
        _cell2 = _fpptd->neighborAddress;
    } else {
        _cell2 = _cell;
        _cell1 = _fpptd->neighborAddress;
    }

    if (!linkInv.getLinkByCells(_cell1, _cell2)) {
        createFocalPointPlasticityLink(_cell1, _cell2, _fpptd->lambdaDistance, _fpptd->targetDistance,
                                       _fpptd->maxDistance);
    }

}

void FocalPointPlasticityPlugin::insertInternalFPPData(CellG *_cell, FocalPointPlasticityTrackerData *_fpptd) {

    CellG *_cell1, *_cell2;
    if (_fpptd->isInitiator) {
        _cell1 = _cell;
        _cell2 = _fpptd->neighborAddress;
    } else {
        _cell2 = _cell;
        _cell1 = _fpptd->neighborAddress;
    }

    if (!linkInvInternal.getLinkByCells(_cell1, _cell2)) {
        createInternalFocalPointPlasticityLink(_cell1, _cell2, _fpptd->lambdaDistance, _fpptd->targetDistance,
                                               _fpptd->maxDistance);
    }

}

void FocalPointPlasticityPlugin::insertAnchorFPPData(CellG *_cell, FocalPointPlasticityTrackerData *_fpptd) {

    int anchorId = _fpptd->anchorId;
    if (!linkInvAnchor.getAnchor(_cell, _fpptd->anchorId)) {
        createAnchor(_cell, _fpptd->lambdaDistance, _fpptd->targetDistance, _fpptd->maxDistance, _fpptd->anchorPoint[0],
                     _fpptd->anchorPoint[1], _fpptd->anchorPoint[2]);
    }

}


std::vector <FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getFPPDataVec(CellG *_cell) {
    std::vector <FocalPointPlasticityTrackerData> dataVec;

    for (auto &link: linkInv.getCellLinkList(_cell)) {
        dataVec.push_back(link->getFPPTrackerData(_cell));
    }


    return dataVec;
}

std::vector <FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getInternalFPPDataVec(CellG *_cell) {
    std::vector <FocalPointPlasticityTrackerData> dataVec;

    for (auto &link: linkInvInternal.getCellLinkList(_cell)) {
        dataVec.push_back(link->getFPPTrackerData(_cell));
    }

    return dataVec;
}

std::vector <FocalPointPlasticityTrackerData> FocalPointPlasticityPlugin::getAnchorFPPDataVec(CellG *_cell) {
    std::vector <FocalPointPlasticityTrackerData> dataVec;

    for (auto &link: linkInvAnchor.getCellLinkList(_cell)) {
        dataVec.push_back(link->getFPPTrackerData(_cell));
    }

    return dataVec;
}

double FocalPointPlasticityPlugin::tryAddingNewJunction(const Point3D &pt, const CellG *newCell) {

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    short &newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
    CellG *&newNeighbor = newNeighborVec[currentWorkNodeNumber];

    auto newCellPlastParamsItr = plastParamsArray.find(newCell->type);

    if (newCellPlastParamsItr == plastParamsArray.end()) { //the newCell type is not listed by the user
        newJunctionInitiatedFlag = false;
        return 0.0;
    }

    auto newCellPlastParams = newCellPlastParamsItr->second;

    //check if new cell can accept new junctions
    if (linkInv.getNumberOfJunctions(const_cast<CellG *>(newCell)) >= maxNumberOfJunctionsTotalMap[newCell->type]) {
        newJunctionInitiatedFlag = false;
        return 0.0;

    }

    boundaryStrategy = BoundaryStrategy::getInstance();

    int maxNeighborIndexLocal = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
    Neighbor neighbor;
    CellG *nCell;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    //visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

    // Randomly shuffle neighbor selection
    std::vector<unsigned int> randNeighborIndex = std::vector<unsigned int>(maxNeighborIndexLocal + 1, 0);
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) randNeighborIndex[nIdx] = nIdx;

    // in a regular run we do randomly select a neighbor with which we make a new link. In test run or test data generate run we do not perform this randomization
    if (!potts->get_test_output_generate_flag() && !potts->get_test_run_flag()) {
        std::random_shuffle(randNeighborIndex.begin(), randNeighborIndex.end());
    }

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), randNeighborIndex[nIdx]);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }
        nCell = fieldG->get(neighbor.pt);

        if (!nCell)
            //no junctions with medium
            continue;


        if (nCell == newCell || nCell->clusterId ==
                                newCell->clusterId)    // make sure that newCell and nCell are different and belong to different clusters
            continue;

        auto plastParamsItr = newCellPlastParams.find(nCell->type);

        if (plastParamsItr == newCellPlastParams.end())
            continue; // check if there are parameters for this combination of cell types

        auto plastParams = plastParamsItr->second;

        // check if neighbor cell can accept another junction
        if (linkInv.getNumberOfJunctionsByType(nCell, newCell->type) >= plastParams.maxNumberOfJunctions) {
            continue;
        }

        // check if new cell can accept another junction
        if (linkInv.getNumberOfJunctionsByType(const_cast<CellG *>(newCell), nCell->type) >=
            plastParams.maxNumberOfJunctions) {
            continue;
        }

        //check if nCell has a junction with newCell
        if (!linkInv.getLinkByCells(const_cast<CellG *>(newCell), nCell)) {
            //new connection allowed
            newJunctionInitiatedFlag = true;
            newNeighbor = nCell;
            break;
        }
    }

    if (newJunctionInitiatedFlag) {
        return newCellPlastParams[newNeighbor->type].activationEnergy;
    } else {
        return 0.0;
    }
}

double FocalPointPlasticityPlugin::tryAddingNewJunctionWithinCluster(const Point3D &pt, const CellG *newCell) {
    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();

    short &newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
    CellG *&newNeighbor = newNeighborVec[currentWorkNodeNumber];

    auto newCellPlastParamsItr = internalPlastParamsArray.find(newCell->type);

    if (newCellPlastParamsItr == internalPlastParamsArray.end()) { //the newCell type is not listed by the user
        newJunctionInitiatedFlagWithinCluster = false;
        return 0.0;
    }

    auto newCellPlastParams = newCellPlastParamsItr->second;

    //check if new cell can accept new junctions
    if (linkInvInternal.getNumberOfJunctions(const_cast<CellG *>(newCell)) >=
        maxNumberOfJunctionsInternalTotalMap[newCell->type]) {
        newJunctionInitiatedFlagWithinCluster = false;
        return 0.0;
    }

    boundaryStrategy = BoundaryStrategy::getInstance();
    int maxNeighborIndexLocal = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
    Neighbor neighbor;
    CellG *nCell;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    //visit point neighbors of newCell and see if within of specified range there is another cell with which newCell can make a junction

    // Randomly shuffle neighbor selection
    std::vector<unsigned int> randNeighborIndex = std::vector<unsigned int>(maxNeighborIndexLocal + 1, 0);
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) randNeighborIndex[nIdx] = nIdx;
    std::random_shuffle(randNeighborIndex.begin(), randNeighborIndex.end());

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexLocal; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), randNeighborIndex[nIdx]);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }
        nCell = fieldG->get(neighbor.pt);

        if (!nCell)
            //no junctions with medium
            continue;

        if (nCell == newCell || nCell->clusterId !=
                                newCell->clusterId)    // make sure that newCell and nCell are different and belong to the same clusters
            continue;

        auto plastParamsItr = newCellPlastParams.find(nCell->type);

        //check if type of neighbor cell is listed by the user
        if (plastParamsItr == newCellPlastParams.end()) continue;

        auto plastParams = plastParamsItr->second;

        if (maxNumberOfJunctionsInternalTotalMap[newCell->type] == 0) {
            continue;
        }

        // check if neighbor cell can accept another junction

        if (linkInvInternal.getNumberOfJunctionsByType(nCell, newCell->type) >= plastParams.maxNumberOfJunctions) {
            continue;
        }

        // check if newCell can accept another junction
        if (linkInvInternal.getNumberOfJunctionsByType(const_cast<CellG *>(newCell), nCell->type) >=
            plastParams.maxNumberOfJunctions) {
            continue;
        }

        //check if nCell has has a junction with newCell

        if (!linkInvInternal.getLinkByCells(const_cast<CellG *>(newCell), nCell)) {
            //new connection allowed
            newJunctionInitiatedFlagWithinCluster = true;
            newNeighbor = nCell;
            break;
        }
    }


    if (newJunctionInitiatedFlagWithinCluster) {
        return newCellPlastParams[newNeighbor->type].activationEnergy;
    } else {
        return 0.0;
    }

}

double FocalPointPlasticityPlugin::changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {

    //This plugin will not work properly with periodic boundary conditions. If necessary I can fix it
    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return 0.0;


    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    short &newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
    short &newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
    CellG *&newNeighbor = newNeighborVec[currentWorkNodeNumber];

    double energy = 0.0;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    Neighbor neighbor;
    Neighbor neighborOfNeighbor;
    CellG *nCell;

    newJunctionInitiatedFlag = false;
    newJunctionInitiatedFlagWithinCluster = false;

    newNeighbor = 0;

    //check if we need to create new junctions only new cell can initiate junctions
    if (newCell) {
        double activationEnergy = tryAddingNewJunctionWithinCluster(pt, newCell);
        if (newJunctionInitiatedFlagWithinCluster) {
            energy += activationEnergy;

            return energy;
        }
    }

    if (newCell) {
        double activationEnergy = tryAddingNewJunction(pt, newCell);
        if (newJunctionInitiatedFlag) {
            energy += activationEnergy;

            return energy;
        }
    }

    Coordinates3D<double> centroidOldAfter;
    Coordinates3D<double> centroidNewAfter;
    Coordinates3D<float> centMassOldAfter;
    Coordinates3D<float> centMassNewAfter;
    Coordinates3D<float> centMassOldBefore;
    Coordinates3D<float> centMassNewBefore;


    if (oldCell) {

        float volOldCell = (float) oldCell->volume;
        centMassOldBefore = Coordinates3D<float>(oldCell->xCM / volOldCell, oldCell->yCM / volOldCell,
                                                 oldCell->zCM / volOldCell);

        if (oldCell->volume > 1) {
            centroidOldAfter = precalculateCentroid(pt, oldCell, -1, fieldDim, boundaryStrategy);
            centMassOldAfter = Coordinates3D<float>(centroidOldAfter.x / (volOldCell - 1),
                                                    centroidOldAfter.y / (volOldCell - 1),
                                                    centroidOldAfter.z / (volOldCell - 1));

        } else {
            centroidOldAfter = Coordinates3D<double>(oldCell->xCM, oldCell->yCM, oldCell->zCM);
            centMassOldAfter = Coordinates3D<float>(centroidOldAfter.x / volOldCell, centroidOldAfter.y / volOldCell,
                                                    centroidOldAfter.z / volOldCell);

        }
    }

    if (newCell) {
        float volNewCell = (float) newCell->volume;

        centMassNewBefore = Coordinates3D<float>(newCell->xCM / volNewCell, newCell->yCM / volNewCell,
                                                 newCell->zCM / volNewCell);
        centroidNewAfter = precalculateCentroid(pt, newCell, 1, fieldDim, boundaryStrategy);
        centMassNewAfter = Coordinates3D<float>(centroidNewAfter.x / (volNewCell + 1),
                                                centroidNewAfter.y / (volNewCell + 1),
                                                centroidNewAfter.z / (volNewCell + 1));
    }

    //will loop over neighbors of the oldCell and calculate Plasticity energy

    std::vector<float> anchorPoint;

    float deltaL;
    float lBefore;
    float nCellVol;
    if (oldCell) {

        for (auto &link: linkInv.getCellLinkList(const_cast<CellG *>(oldCell))) {

            nCell = link->getOtherCell(oldCell);
            nCellVol = nCell->volume;

            if (nCell != newCell) {
                lBefore = distInvariantCM(centMassOldBefore.X(), centMassOldBefore.Y(), centMassOldBefore.Z(),
                                          nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassOldAfter.X(), centMassOldAfter.Y(), centMassOldAfter.Z(),
                                        nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                        boundaryStrategy)
                        - lBefore;
            } else {
                lBefore = distInvariantCM(centMassOldBefore.X(), centMassOldBefore.Y(), centMassOldBefore.Z(),
                                          centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(), fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassOldAfter.X(), centMassOldAfter.Y(), centMassOldAfter.Z(),
                                        centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(), fieldDim,
                                        boundaryStrategy)
                        - lBefore;
            }
            energy += (this->*diffEnergyFcnPtr)(deltaL, lBefore, link->getFPPTrackerData(const_cast<CellG *>(oldCell)),
                                                oldCell, false);
        }

        //go over compartments
        for (auto &link: linkInvInternal.getCellLinkList(const_cast<CellG *>(oldCell))) {
            nCell = link->getOtherCell(oldCell);
            nCellVol = nCell->volume;

            if (nCell != newCell) {
                lBefore = distInvariantCM(centMassOldBefore.X(), centMassOldBefore.Y(), centMassOldBefore.Z(),
                                          nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassOldAfter.X(), centMassOldAfter.Y(), centMassOldAfter.Z(),
                                        nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                        boundaryStrategy)
                        - lBefore;
            } else {
                lBefore = distInvariantCM(centMassOldBefore.X(), centMassOldBefore.Y(), centMassOldBefore.Z(),
                                          centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(), fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassOldAfter.X(), centMassOldAfter.Y(), centMassOldAfter.Z(),
                                        centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(), fieldDim,
                                        boundaryStrategy)
                        - lBefore;
            }
            double clusterOldCellEnergy = (this->*diffEnergyFcnPtr)(deltaL, lBefore, link->getFPPTrackerData(
                    const_cast<CellG *>(oldCell)), oldCell, true);
            energy += clusterOldCellEnergy;
        }

        //go over anchors
        for (auto &link: linkInvAnchor.getCellLinkList(const_cast<CellG *>(oldCell))) {
            anchorPoint = link->getAnchorPoint();
            lBefore = distInvariantCM(centMassOldBefore.X(), centMassOldBefore.Y(), centMassOldBefore.Z(),
                                      anchorPoint[0], anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy);
            deltaL =
                    distInvariantCM(centMassOldAfter.X(), centMassOldAfter.Y(), centMassOldAfter.Z(), anchorPoint[0],
                                    anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy)
                    - lBefore;
            energy += (this->*diffEnergyFcnPtr)(deltaL, lBefore, link->getFPPTrackerData(const_cast<CellG *>(oldCell)),
                                                oldCell, false);
        }

    }

    if (newCell) {

        for (auto &link: linkInv.getCellLinkList(const_cast<CellG *>(newCell))) {
            nCell = link->getOtherCell(newCell);
            nCellVol = nCell->volume;

            if (nCell != oldCell) {
                lBefore = distInvariantCM(centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(),
                                          nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(),
                                        nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                        boundaryStrategy)
                        - lBefore;

                double newDeltaEnergy = (this->*diffEnergyFcnPtr)(deltaL, lBefore,
                                                                  link->getFPPTrackerData(const_cast<CellG *>(newCell)),
                                                                  newCell, false);
                energy += newDeltaEnergy;

            } else {
                // this was already taken into account in the oldCell secion - we need to avoid double counting
            }
        }

        //go ever compartments
        for (auto &link: linkInvInternal.getCellLinkList(const_cast<CellG *>(newCell))) {
            nCell = link->getOtherCell(newCell);
            nCellVol = nCell->volume;

            if (nCell != oldCell) {
                lBefore = distInvariantCM(centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(),
                                          nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                          boundaryStrategy);
                deltaL =
                        distInvariantCM(centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(),
                                        nCell->xCM / nCellVol, nCell->yCM / nCellVol, nCell->zCM / nCellVol, fieldDim,
                                        boundaryStrategy)
                        - lBefore;
                energy += (this->*diffEnergyFcnPtr)(deltaL, lBefore,
                                                    link->getFPPTrackerData(const_cast<CellG *>(newCell)), newCell,
                                                    true);
            } else {
                // this was already taken into account in the oldCell secion - we need to avoid double counting
            }
        }

        //go over anchors
        for (auto &link: linkInvAnchor.getCellLinkList(const_cast<CellG *>(newCell))) {
            anchorPoint = link->getAnchorPoint();
            lBefore = distInvariantCM(centMassNewBefore.X(), centMassNewBefore.Y(), centMassNewBefore.Z(),
                                      anchorPoint[0], anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy);
            deltaL =
                    distInvariantCM(centMassNewAfter.X(), centMassNewAfter.Y(), centMassNewAfter.Z(), anchorPoint[0],
                                    anchorPoint[1], anchorPoint[2], fieldDim, boundaryStrategy)
                    - lBefore;
            energy += (this->*diffEnergyFcnPtr)(deltaL, lBefore, link->getFPPTrackerData(const_cast<CellG *>(newCell)),
                                                newCell, false);
        }

    }
    return energy;
}


void FocalPointPlasticityPlugin::deleteFocalPointPlasticityLink(CellG *_cell1, CellG *_cell2) {

    FocalPointPlasticityLink *link = linkInv.getLinkByCells(_cell1, _cell2);
    if (link) linkInv.removeFromInventory(link);

}

void FocalPointPlasticityPlugin::deleteInternalFocalPointPlasticityLink(CellG *_cell1, CellG *_cell2) {

    FocalPointPlasticityInternalLink *link = linkInvInternal.getLinkByCells(_cell1, _cell2);
    if (link) linkInvInternal.removeFromInventory(link);

}

void FocalPointPlasticityPlugin::createFocalPointPlasticityLink(CellG *_cell1, CellG *_cell2, double _lambda,
                                                                double _targetDistance, double _maxDistance) {

    FocalPointPlasticityLinkTrackerData fpptld = FocalPointPlasticityLinkTrackerData(
            plastParamsArray[_cell1->type][_cell2->type]);
    fpptld.lambdaDistance = _lambda;
    fpptld.targetDistance = _targetDistance;
    fpptld.maxDistance = _maxDistance;
    fpptld.initMCS = sim->getStep();
    FocalPointPlasticityLink *link = new FocalPointPlasticityLink(_cell1, _cell2, potts, fpptld);
    linkInv.addToInventory(link);

}

void FocalPointPlasticityPlugin::createInternalFocalPointPlasticityLink(CellG *_cell1, CellG *_cell2, double _lambda,
                                                                        double _targetDistance, double _maxDistance) {

    FocalPointPlasticityLinkTrackerData fpptld = FocalPointPlasticityLinkTrackerData(
            internalPlastParamsArray[_cell1->type][_cell2->type]);
    fpptld.lambdaDistance = _lambda;
    fpptld.targetDistance = _targetDistance;
    fpptld.maxDistance = _maxDistance;
    fpptld.initMCS = sim->getStep();
    FocalPointPlasticityInternalLink *link = new FocalPointPlasticityInternalLink(_cell1, _cell2, potts, fpptld);
    linkInvInternal.addToInventory(link);

}


void FocalPointPlasticityPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    int currentWorkNodeNumber = pUtils->getCurrentWorkNodeNumber();
    short &newJunctionInitiatedFlag = newJunctionInitiatedFlagVec[currentWorkNodeNumber];
    short &newJunctionInitiatedFlagWithinCluster = newJunctionInitiatedFlagWithinClusterVec[currentWorkNodeNumber];
    CellG *&newNeighbor = newNeighborVec[currentWorkNodeNumber];

    if (oldCell && oldCell->volume == 0) {

#pragma omp critical
        {

            linkInv.removeCellLinks(oldCell);

            //go over compartments
            linkInvInternal.removeCellLinks(oldCell);

            //go over anchors
            linkInvAnchor.removeCellLinks(oldCell);
        }
    }

    //because newJunctionInitiatedFlag is in principle "global" variable changing lattice configuration from python e.g. delete cell
    // can cause newJunctionInitiatedFlag be still after last change and thus falsly indicate new junction even when cell is e.g. Medium

    if (newJunctionInitiatedFlag && newCell) {

        // we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects
        newJunctionInitiatedFlag = false;

        double xCMNew = newCell->xCM / float(newCell->volume);
        double yCMNew = newCell->yCM / float(newCell->volume);
        double zCMNew = newCell->zCM / float(newCell->volume);

        double xCMNeighbor = newNeighbor->xCM / float(newNeighbor->volume);
        double yCMNeighbor = newNeighbor->yCM / float(newNeighbor->volume);
        double zCMNeighbor = newNeighbor->zCM / float(newNeighbor->volume);

        double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim,
                                          boundaryStrategy);

        if (functionType == BYCELLTYPE || functionType == BYCELLID) {
            FocalPointPlasticityTrackerData fpptd = plastParamsArray[newCell->type][newNeighbor->type];
            createFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance,
                                           fpptd.maxDistance);

        } else if (functionType == GLOBAL) {
            FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(newNeighbor, lambda,
                                                                                    0.9 * distance);
            createFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance,
                                           fpptd.maxDistance);

        }
        //}
        return;
    }

    //because newJunctionInitiatedFlag is in principle "global" variable changing lattice configuration from python e.g. delete cell
    // can cause newJunctionInitiatedFlag be still after last change and thus falsly indicate new junction even when cell is e.g. Medium

    if (newJunctionInitiatedFlagWithinCluster && newCell) {

        // we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects
        newJunctionInitiatedFlagWithinCluster = false;

        double xCMNew = newCell->xCM / float(newCell->volume);
        double yCMNew = newCell->yCM / float(newCell->volume);
        double zCMNew = newCell->zCM / float(newCell->volume);

        double xCMNeighbor = newNeighbor->xCM / float(newNeighbor->volume);
        double yCMNeighbor = newNeighbor->yCM / float(newNeighbor->volume);
        double zCMNeighbor = newNeighbor->zCM / float(newNeighbor->volume);

        double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor, fieldDim,
                                          boundaryStrategy);

        if (functionType == BYCELLTYPE || functionType == BYCELLID) {
            FocalPointPlasticityTrackerData fpptd = internalPlastParamsArray[newCell->type][newNeighbor->type];
            createInternalFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance,
                                                   fpptd.maxDistance);

        } else if (functionType == GLOBAL) {
            FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(newNeighbor, lambda,
                                                                                    0.9 * distance);
            createInternalFocalPointPlasticityLink(newCell, newNeighbor, fpptd.lambdaDistance, fpptd.targetDistance,
                                                   fpptd.maxDistance);

        }
        return;
    }

    // we reset the flags here to avoid  keeping true value in the "global" class-wide variable- this might have ide effects
    newJunctionInitiatedFlag = false;
    newJunctionInitiatedFlagWithinCluster = false;

    CellG *nCell;

    if (newCell) {
        double xCMNew = newCell->xCM / float(newCell->volume);
        double yCMNew = newCell->yCM / float(newCell->volume);
        double zCMNew = newCell->zCM / float(newCell->volume);

        //we remove only one cell at a time even though we could do it for many cells
        bool linkRemoved = false;
        std::vector<float> anchorPoint;
        //we will first remove anchor links if they fit removal criteria
#pragma omp critical
        {
            for (auto &link: linkInvAnchor.getCellLinkList(newCell)) {
                anchorPoint = link->getAnchorPoint();

                double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, anchorPoint[0], anchorPoint[1],
                                                  anchorPoint[2], fieldDim, boundaryStrategy);
                int maxDistanceLocal;
                if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                    maxDistanceLocal = link->getMaxDistance();
                } else if (functionType == GLOBAL) {
                    maxDistanceLocal = maxDistance;
                }

                if (distance > maxDistanceLocal) {

                    deleteAnchor(newCell, link->getAnchorId());
                    linkRemoved = true;
                    break;
                }
            }

            if (!linkRemoved) {
                for (auto &link: linkInv.getCellLinkList(newCell)) {
                    nCell = link->getOtherCell(newCell);

                    double xCMNeighbor = nCell->xCM / float(nCell->volume);
                    double yCMNeighbor = nCell->yCM / float(nCell->volume);
                    double zCMNeighbor = nCell->zCM / float(nCell->volume);
                    double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor,
                                                      fieldDim, boundaryStrategy);
                    int maxDistanceLocal;
                    if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                        maxDistanceLocal = link->getMaxDistance();
                    } else if (functionType == GLOBAL) {
                        maxDistanceLocal = maxDistance;
                    }

                    if (distance > maxDistanceLocal) {
                        deleteFocalPointPlasticityLink(newCell, nCell);
                        break;
                    }
                }
            }

        }

        //go over compartments
#pragma omp critical
        {
            for (auto &link: linkInvInternal.getCellLinkList(newCell)) {
                nCell = link->getOtherCell(newCell);
                //we remove only one cell at a time even though we could do it for many cells
                double xCMNeighbor = nCell->xCM / float(nCell->volume);
                double yCMNeighbor = nCell->yCM / float(nCell->volume);
                double zCMNeighbor = nCell->zCM / float(nCell->volume);
                double distance = distInvariantCM(xCMNew, yCMNew, zCMNew, xCMNeighbor, yCMNeighbor, zCMNeighbor,
                                                  fieldDim, boundaryStrategy);
                int maxDistanceLocal;
                if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                    maxDistanceLocal = link->getMaxDistance();
                } else if (functionType == GLOBAL) {
                    maxDistanceLocal = maxDistance;
                }

                if (distance > maxDistanceLocal) {
                    deleteInternalFocalPointPlasticityLink(newCell, nCell);
                    break;
                }

            }

        }
    }

    if (oldCell) {
        double xCMOld = oldCell->xCM / float(oldCell->volume);
        double yCMOld = oldCell->yCM / float(oldCell->volume);
        double zCMOld = oldCell->zCM / float(oldCell->volume);
        //we remove only one cell at a time even though we could do it for many cells
        bool linkRemoved = false;
        std::vector<float> anchorPoint;
        //we will first remove anchor links if they fit removal criteria
#pragma omp critical
        {
            for (auto &link: linkInvAnchor.getCellLinkList(oldCell)) {
                anchorPoint = link->getAnchorPoint();

                double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, anchorPoint[0], anchorPoint[1],
                                                  anchorPoint[2], fieldDim, boundaryStrategy);
                int maxDistanceLocal;
                if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                    maxDistanceLocal = link->getMaxDistance();
                } else if (functionType == GLOBAL) {
                    maxDistanceLocal = maxDistance;
                }

                if (distance > maxDistanceLocal) {
                    deleteAnchor(oldCell, link->getAnchorId());
                    linkRemoved = true;
                    break;
                }

            }

            if (!linkRemoved) {
                for (auto &link: linkInv.getCellLinkList(oldCell)) {
                    nCell = link->getOtherCell(oldCell);

                    double xCMNeighbor = nCell->xCM / float(nCell->volume);
                    double yCMNeighbor = nCell->yCM / float(nCell->volume);
                    double zCMNeighbor = nCell->zCM / float(nCell->volume);
                    double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, xCMNeighbor, yCMNeighbor, zCMNeighbor,
                                                      fieldDim, boundaryStrategy);
                    //double distance=dist(xCMOld,yCMOld,zCMOld,xCMNeighbor,yCMNeighbor,zCMNeighbor);
                    int maxDistanceLocal;
                    if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                        maxDistanceLocal = link->getMaxDistance();
                    } else if (functionType == GLOBAL) {
                        maxDistanceLocal = maxDistance;
                    }

                    if (distance > maxDistanceLocal) {
                        deleteFocalPointPlasticityLink(oldCell, nCell);
                        break;
                    }
                }

            }

        }

        //go over compartments
#pragma omp critical
        {
            for (auto &link: linkInvInternal.getCellLinkList(oldCell)) {
                nCell = link->getOtherCell(oldCell);
                //we remove only one cell at a time even though we could do it for many cells many cells
                double xCMNeighbor = nCell->xCM / float(nCell->volume);
                double yCMNeighbor = nCell->yCM / float(nCell->volume);
                double zCMNeighbor = nCell->zCM / float(nCell->volume);
                double distance = distInvariantCM(xCMOld, yCMOld, zCMOld, xCMNeighbor, yCMNeighbor, zCMNeighbor,
                                                  fieldDim, boundaryStrategy);
                //double distance=dist(xCMOld,yCMOld,zCMOld,xCMNeighbor,yCMNeighbor,zCMNeighbor);
                int maxDistanceLocal;
                if (functionType == BYCELLTYPE || functionType == BYCELLID) {
                    maxDistanceLocal = link->getMaxDistance();
                } else if (functionType == GLOBAL) {
                    maxDistanceLocal = maxDistance;
                }

                if (distance > maxDistanceLocal) {
                    deleteInternalFocalPointPlasticityLink(oldCell, nCell);
                    break;
                }
            }

        }

    }

}

void FocalPointPlasticityPlugin::setFocalPointPlasticityParameters(CellG *_cell1, CellG *_cell2, double _lambda,
                                                                   double _targetDistance, double _maxDistance) {

    FocalPointPlasticityLink *link = linkInv.getLinkByCells(_cell1, _cell2);
    if (link) {
        link->setLambdaDistance(_lambda);
        if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
        if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);
    }
}

void FocalPointPlasticityPlugin::setInternalFocalPointPlasticityParameters(CellG *_cell1, CellG *_cell2, double _lambda,
                                                                           double _targetDistance,
                                                                           double _maxDistance) {

    FocalPointPlasticityInternalLink *link = linkInvInternal.getLinkByCells(_cell1, _cell2);
    if (link) {
        link->setLambdaDistance(_lambda);
        if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
        if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);
    }
}

double FocalPointPlasticityPlugin::getPlasticityParametersLambdaDistance(CellG *_cell1, CellG *_cell2) {

    FocalPointPlasticityLink *link = linkInv.getLinkByCells(_cell1, _cell2);
    if (link) {
        return link->getLambdaDistance();
    } else {
        return 0.0;
    }
}

double FocalPointPlasticityPlugin::getPlasticityParametersTargetDistance(CellG *_cell1, CellG *_cell2) {

    FocalPointPlasticityLink *link = linkInv.getLinkByCells(_cell1, _cell2);
    if (link) {
        return link->getTargetDistance();
    } else {
        return 0.0;
    }
}

int FocalPointPlasticityPlugin::createAnchor(CellG *_cell, double _lambda, double _targetDistance, double _maxDistance,
                                             float _x, float _y, float _z) {

    int newAnchorId = linkInvAnchor.getNextAnchorId(_cell);
    FocalPointPlasticityLinkTrackerData fppltd(_lambda, _targetDistance, _maxDistance, sim->getStep());
    fppltd.anchorPoint[0] = _x;
    fppltd.anchorPoint[1] = _y;
    fppltd.anchorPoint[2] = _z;
    fppltd.anchorId = newAnchorId;
    FocalPointPlasticityAnchor *link = new FocalPointPlasticityAnchor(_cell, potts, fppltd);
    linkInvAnchor.addToInventory(link);

    return newAnchorId;

}

void FocalPointPlasticityPlugin::deleteAnchor(CellG *_cell, int _anchorId) {

    FocalPointPlasticityAnchor *link = linkInvAnchor.getAnchor(_cell, _anchorId);
    if (link) linkInvAnchor.removeFromInventory(link);

}

void
FocalPointPlasticityPlugin::setAnchorParameters(CellG *_cell, int _anchorId, double _lambda, double _targetDistance,
                                                double _maxDistance, float _x, float _y, float _z) {

    FocalPointPlasticityAnchor *link = linkInvAnchor.getAnchor(_cell, _anchorId);
    if (link) {
        link->setLambdaDistance(_lambda);
        if (_targetDistance != 0.0) link->setTargetDistance(_targetDistance);
        if (_maxDistance != 0.0) link->setMaxDistance(_maxDistance);

        std::vector<float> anchorPoint(link->getAnchorPoint());
        if (_x != -1) anchorPoint[0] = _x;
        if (_y != -1) anchorPoint[1] = _y;
        if (_z != -1) anchorPoint[2] = _z;
        link->setAnchorPoint(anchorPoint);
    }

}

std::string FocalPointPlasticityPlugin::steerableName() { return "FocalPointPlasticity"; }

std::string FocalPointPlasticityPlugin::toString() { return steerableName(); }
