#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>

using namespace std;

#include "ContactLocalFlexPlugin.h"
#include <Logger/CC3DLogger.h>

ContactLocalFlexPlugin::ContactLocalFlexPlugin() :
        pUtils(0),
        lockPtr(0),
        depth(1),
        weightDistance(false),
        boundaryStrategy(0),
        xmlData(0) {
    initializadContactData = false;
}

ContactLocalFlexPlugin::~ContactLocalFlexPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void ContactLocalFlexPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    potts->getCellFactoryGroupPtr()->registerClass(&contactDataContainerAccessor);


    bool pluginAlreadyRegisteredFlag;
    //this will load SurfaceTracker plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(sim);

    potts->registerEnergyFunction(this);
    potts->registerCellGChangeWatcher(this);
    simulator->registerSteerableObject(this);
}

void ContactLocalFlexPlugin::extraInit(Simulator *simulator) {
    update(xmlData, true);
}

double ContactLocalFlexPlugin::changeEnergy(const Point3D &pt,
    const CellG *newCell,
    const CellG *oldCell) {

    double energy = 0;
    Point3D n;

    CellG *nCell = 0;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    Neighbor neighbor;


    if (weightDistance) {
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }
            nCell = fieldG->get(neighbor.pt);
            if (nCell != oldCell) {
                if ((nCell != 0) && (oldCell != 0)) {
                    if ((nCell->clusterId) != (oldCell->clusterId)) {
                        energy -= contactEnergy(oldCell, nCell) / neighbor.distance;
                    }
                } else {
                    energy -= contactEnergy(oldCell, nCell) / neighbor.distance;
                }
            }
            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += contactEnergy(newCell, nCell) / neighbor.distance;
                    }
                } else {
                    energy += contactEnergy(newCell, nCell) / neighbor.distance;

                }
            }


        }

    } else {
        //default behaviour  no energy weighting 
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            nCell = fieldG->get(neighbor.pt);
            if (nCell != oldCell) {
                if ((nCell != 0) && (oldCell != 0)) {
                    if ((nCell->clusterId) != (oldCell->clusterId)) {
                        energy -= contactEnergy(oldCell, nCell);
                    }
                } else {
                    energy -= contactEnergy(oldCell, nCell);
                }
            }

            if (nCell != newCell) {
                if ((newCell != 0) && (nCell != 0)) {
                    if ((newCell->clusterId) != (nCell->clusterId)) {
                        energy += contactEnergy(newCell, nCell);
                    }
                } else {
                    energy += contactEnergy(newCell, nCell);

                }

            }


        }

    }

    return energy;
}

void ContactLocalFlexPlugin::setContactEnergy(const string typeName1,
                                              const string typeName2,
                                              const double energy) {

    unsigned char type1 = automaton->getTypeId(typeName1);
    unsigned char type2 = automaton->getTypeId(typeName2);

    contactEnergyArray[type1][type2] = energy;
    contactEnergyArray[type2][type1] = energy;
}

int ContactLocalFlexPlugin::getIndex(const int type1, const int type2) const {
    if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
    else return ((type2 + 1) | ((type1 + 1) << 16));
}


double ContactLocalFlexPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
    ContactLocalFlexData clfdObj;
    CellG *cell;
    CellG *neighbor;

    if (cell1) {
        cell = const_cast<CellG *>(cell1);
        neighbor = const_cast<CellG *>(cell2);
    } else {
        cell = const_cast<CellG *>(cell2);
        neighbor = const_cast<CellG *>(cell1);
    }

    set <ContactLocalFlexData> &clfdSet = contactDataContainerAccessor.get(cell->extraAttribPtr)->contactDataContainer;


    clfdObj.neighborAddress = neighbor;

    set<ContactLocalFlexData>::iterator sitrCD = clfdSet.find(clfdObj);

    if (sitrCD != clfdSet.end()) {
        return sitrCD->J;
    } else {
        return defaultContactEnergy(cell1, cell2);
    }

}

double ContactLocalFlexPlugin::defaultContactEnergy(const CellG *cell1, const CellG *cell2) {
    //implementing only referring to the local defaultContactEnergy
    return contactEnergyArray[cell1 ? cell1->type : 0][cell2 ? cell2->type : 0];
}


//this function is called once per simulation after cells have been assigned types
// (some initializers postpone type initialization)
void ContactLocalFlexPlugin::initializeContactLocalFlexData() {

    //we double-check this flag to make sure this function does not get called multiple times by different threads
    if (initializadContactData)
        return;

    CellInventory *cellInventoryPtr = &potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;
    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);

        ContactLocalFlexDataContainer *dataContainer = contactDataContainerAccessor.get(cell->extraAttribPtr);
        dataContainer->localDefaultContactEnergies = contactEnergyArray;
    }


    for (cInvItr = cellInventoryPtr->cellInventoryBegin(); cInvItr != cellInventoryPtr->cellInventoryEnd(); ++cInvItr) {
        cell = cellInventoryPtr->getCell(cInvItr);

        set <ContactLocalFlexData> &clfdSet = contactDataContainerAccessor.get(
                cell->extraAttribPtr)->contactDataContainer;
        clfdSet.clear();
        updateContactEnergyData(cell);
    }

    initializadContactData = true;
}

void ContactLocalFlexPlugin::updateContactEnergyData(CellG *_cell) {
    //this function syncs neighbor list for _cell and contact data for _cell so that they contain same number of corresponding
    //entries

    NeighborTrackerPlugin *neighborTrackerPlugin = (NeighborTrackerPlugin *) Simulator::pluginManager.get(
            "NeighborTracker");
    ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr = neighborTrackerPlugin->getNeighborTrackerAccessorPtr();
    unsigned int size1 = 0, size2 = 0;

    set <ContactLocalFlexData> &clfdSet = contactDataContainerAccessor.get(_cell->extraAttribPtr)->contactDataContainer;
    set <NeighborSurfaceData> &nsdSet = neighborTrackerAccessorPtr->get(_cell->extraAttribPtr)->cellNeighbors;

    size1 = clfdSet.size();
    size2 = nsdSet.size();


    //if sizes of sets are different then we add any new neighbors from nsdSet and remove those neighbors from clfdSet
    //that do not show up in nsdSet anymore
    //This way we avoid all sorts of problems associated with various configuration of neighbors after spin flip.
    // Although it is not the fastest algorithm , it is very simple and self explanatory and given the fact that in most
    // cases number of neighbors is fairly small all those inefficiencies do not matter too much.

    ContactLocalFlexData clfdObj;
    NeighborSurfaceData nfdObj;
    set<NeighborSurfaceData>::iterator sitrND;
    set<ContactLocalFlexData>::iterator sitrCD;

    //here we insert neighbors from nsdSet into clfdSet that do not show up in clfdSet
    for (sitrND = nsdSet.begin(); sitrND != nsdSet.end(); ++sitrND) {
        clfdObj.neighborAddress = sitrND->neighborAddress;
        clfdObj.J = defaultContactEnergy(clfdObj.neighborAddress, _cell);


        clfdSet.insert(clfdObj); //the element will be inserted only if it is not there
    }

    //here we remove neighbors from clfd if they do not show up in nsdSet
    for (sitrCD = clfdSet.begin();
         sitrCD != clfdSet.end();) { //notice that incrementing takes place in the loop because we are erasing elements
        nfdObj.neighborAddress = sitrCD->neighborAddress;
        sitrND = nsdSet.find(nfdObj);
        if (sitrND == nsdSet.end()) { //did not find nfdObj.neighborAddress in nsdSet  - need to remove it from clfdSet 
            clfdSet.erase(sitrCD++);
        } else {
            ++sitrCD;
        }
    }

}

void ContactLocalFlexPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
    if (!initializadContactData && sim->getStep() == 0) {
        pUtils->setLock(lockPtr);
        initializeContactLocalFlexData();
        pUtils->unsetLock(lockPtr);
    }

    if (newCell) {
        updateContactEnergyData(newCell);
    }
    if (oldCell) {
        updateContactEnergyData(oldCell);
    }
}


void ContactLocalFlexPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    set<unsigned char> cellTypesSet;
    contactEnergyArray.clear();

    CC3DXMLElementList energyVec = _xmlData->getElements("Energy");

    for (int i = 0; i < energyVec.size(); ++i) {

        setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"),
                         energyVec[i]->getDouble());

        //inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
        cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

    }

    CC3D_Log(LOG_DEBUG) << "size=" << contactEnergyArray.size();
    for (auto &i: cellTypesSet)
        for (auto &j: cellTypesSet) {

            CC3D_Log(LOG_DEBUG) << "contact[" << to_string(i) << "][" << to_string(j) << "]=" << contactEnergyArray[i][j];

        }

    //Here I initialize max neighbor index for direct acces to the list of neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = 0;

    if (_xmlData->getFirstElement("WeightEnergyByDistance")) {
        weightDistance = true;
    }

    if (_xmlData->getFirstElement("Depth")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(
                _xmlData->getFirstElement("Depth")->getDouble());
    } else {
        if (_xmlData->getFirstElement("NeighborOrder")) {

            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                    _xmlData->getFirstElement("NeighborOrder")->getUInt());
        } else {
            maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

        }

    }
    CC3D_Log(LOG_DEBUG) << "Contact maxNeighborIndex=" << maxNeighborIndex;

}


std::string ContactLocalFlexPlugin::toString() {
    return "ContactLocalFlex";
}


std::string ContactLocalFlexPlugin::steerableName() {
    return toString();
}
