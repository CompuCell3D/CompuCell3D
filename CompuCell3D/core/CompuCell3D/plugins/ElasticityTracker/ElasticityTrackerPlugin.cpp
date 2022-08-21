#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;
using namespace std;


#include "ElasticityTrackerPlugin.h"


ElasticityTrackerPlugin::ElasticityTrackerPlugin() :
        pUtils(0),
        lockPtr(0),
        cellFieldG(0),
        initialized(false),
        maxNeighborIndex(0),
        boundaryStrategy(0),
        manualInit(false),
        xmlData(0) {}

ElasticityTrackerPlugin::~ElasticityTrackerPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}


void ElasticityTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

    xmlData = _xmlData;
    simulator = _simulator;
    Potts3D *potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    pUtils = simulator->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    ///getting cell inventory
    cellInventoryPtr = &potts->getCellInventory();



    ///will register ElasticityTracker here
    ExtraMembersGroupAccessorBase *elasticityTrackerAccessorPtr = &elasticityTrackerAccessor;
    ///************************************************************************************************
    ///REMARK. HAVE TO USE THE SAME CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
    ///************************************************************************************************
    potts->getCellFactoryGroupPtr()->registerClass(elasticityTrackerAccessorPtr);


    fieldDim = cellFieldG->getDim();

    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//1st nearest neighbor

    bool pluginAlreadyRegisteredFlag;
    //this will load COM plugin if it is not already loaded
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                  &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);


    //register elasticityTracker after CenterOfMass and after VolumeTracker - implicitly called from CenterOfmass
    potts->registerCellGChangeWatcher(
            this);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ElasticityTrackerPlugin::extraInit(Simulator *_simulator) {
    elasticityTypesNames.clear();
    elasticityTypes.clear();
    CC3DXMLElementList includeTypeNamesXMLVec = xmlData->getElements("IncludeType");
    for (int i = 0; i < includeTypeNamesXMLVec.size(); ++i) {
        elasticityTypesNames.insert(includeTypeNamesXMLVec[i]->getText());
    }

    Automaton *automaton = simulator->getPotts()->getAutomaton();
    // Initializing set of elasticitytypes
    for (set<string>::iterator sitr = elasticityTypesNames.begin(); sitr != elasticityTypesNames.end(); ++sitr) {
        elasticityTypes.insert(automaton->getTypeId(*sitr));
    }

    if (xmlData->getFirstElement("ManualInitialization")) {
        manualInit = true;
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ElasticityTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,
                                            CellG *oldCell) {

    //do not do any updates until the lattice is fully initialized


    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return;

    if (simulator->getStep() >= 0 && !initialized) {
        pUtils->setLock(lockPtr);
        if (!manualInit) {
            initializeElasticityNeighborList(); //we will check  initialized flag inside initializeElasticityNeighborList to make sure that only one thread executes this function - there will be many threads that will enter this section
        }
        pUtils->unsetLock(lockPtr);
    }

    if (initialized) {
        if (oldCell && oldCell->volume == 0 && elasticityTypes.find(oldCell->type) != elasticityTypes.end()) {
            //remove oldCell from neighbor list of old cell neighbors
            set<ElasticityTrackerData>::iterator sitr;
            set <ElasticityTrackerData> *elasticityNeighborsPtr = &elasticityTrackerAccessor.get(
                    oldCell->extraAttribPtr)->elasticityNeighbors;
            set <ElasticityTrackerData> *elasticityNeighborsTmpPtr;
            for (sitr = elasticityNeighborsPtr->begin(); sitr != elasticityNeighborsPtr->end(); ++sitr) {
                //getting set of elasticityNeighbors from the neighbor (pointed by sitr) of the oldCell
                elasticityNeighborsTmpPtr = &elasticityTrackerAccessor.get(
                        sitr->neighborAddress->extraAttribPtr)->elasticityNeighbors;
                //removing oldCell from the set
                elasticityNeighborsTmpPtr->erase(ElasticityTrackerData(oldCell));
            }
        }
    }

}


void ElasticityTrackerPlugin::initializeElasticityNeighborList() {

    //we double-check this flag to makes sure this function does not get called multiple times by different threads
    if (initialized)
        return;
    Point3D pt;
    CellG *cell;
    CellG *nCell;
    Neighbor neighbor;
    set <ElasticityTrackerData> *elasticityNeighborsTmpPtr;
    set<unsigned char>::iterator endSitr = elasticityTypes.end();

    for (unsigned int x = 0; x < fieldDim.x; ++x)
        for (unsigned int y = 0; y < fieldDim.y; ++y)
            for (unsigned int z = 0; z < fieldDim.z; ++z) {
                pt = Point3D(x, y, z);
                cell = cellFieldG->get(pt);
                for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
                    neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
                    if (!neighbor.distance) {
                        //if distance is 0 then the neighbor returned is invalid
                        continue;
                    }
                    nCell = cellFieldG->get(neighbor.pt);
                    if (nCell != cell) {
                        if (nCell &&
                            cell) {//only cells which are of certain types are considered as elasticityNeighbors
                            if (elasticityTypes.find(nCell->type) != endSitr &&
                                elasticityTypes.find(cell->type) != endSitr) {
                                elasticityNeighborsTmpPtr = &elasticityTrackerAccessor.get(
                                        nCell->extraAttribPtr)->elasticityNeighbors;
                                elasticityNeighborsTmpPtr->insert(ElasticityTrackerData(cell));
                                elasticityNeighborsTmpPtr = &elasticityTrackerAccessor.get(
                                        cell->extraAttribPtr)->elasticityNeighbors;
                                elasticityNeighborsTmpPtr->insert(ElasticityTrackerData(nCell));
                            }
                        }
                    }
                }
            }

    initialized = true;

}

void ElasticityTrackerPlugin::assignElasticityPair(CellG *_cell1, CellG *_cell2) {

    if (_cell1 && _cell2) {
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr1;
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr2;
        elasticityNeighborsTmpPtr1 = &elasticityTrackerAccessor.get(_cell1->extraAttribPtr)->elasticityNeighbors;
        elasticityNeighborsTmpPtr1->insert(ElasticityTrackerData(_cell2));

        elasticityNeighborsTmpPtr2 = &elasticityTrackerAccessor.get(_cell2->extraAttribPtr)->elasticityNeighbors;
        elasticityNeighborsTmpPtr2->insert(ElasticityTrackerData(_cell1));
    }

}

void ElasticityTrackerPlugin::addNewElasticLink(CellG *_cell1, CellG *_cell2, float _lambdaElasticityLink,
                                                float _targetLinkLength) {
    if (_cell1 && _cell2) {
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr1;
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr2;
        elasticityNeighborsTmpPtr1 = &elasticityTrackerAccessor.get(_cell1->extraAttribPtr)->elasticityNeighbors;

        elasticityNeighborsTmpPtr1->insert(ElasticityTrackerData(_cell2, _lambdaElasticityLink, _targetLinkLength));

        elasticityNeighborsTmpPtr2 = &elasticityTrackerAccessor.get(_cell2->extraAttribPtr)->elasticityNeighbors;
        elasticityNeighborsTmpPtr2->insert(ElasticityTrackerData(_cell1, _lambdaElasticityLink, _targetLinkLength));
    }
}


void ElasticityTrackerPlugin::removeElasticityPair(CellG *_cell1, CellG *_cell2) {

    if (_cell1 && _cell2) {
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr1;
        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr2;
        elasticityNeighborsTmpPtr1 = &elasticityTrackerAccessor.get(_cell1->extraAttribPtr)->elasticityNeighbors;
        elasticityNeighborsTmpPtr1->erase(ElasticityTrackerData(_cell2));

        elasticityNeighborsTmpPtr2 = &elasticityTrackerAccessor.get(_cell2->extraAttribPtr)->elasticityNeighbors;
        elasticityNeighborsTmpPtr2->erase(ElasticityTrackerData(_cell1));
    }


}

ElasticityTrackerData *ElasticityTrackerPlugin::findTrackerData(CellG *_cell1, CellG *_cell2) {
    if (_cell1 && _cell2) {

        set <ElasticityTrackerData> *elasticityNeighborsTmpPtr1;
        set<ElasticityTrackerData>::iterator sitr;

        elasticityNeighborsTmpPtr1 = &elasticityTrackerAccessor.get(_cell1->extraAttribPtr)->elasticityNeighbors;
        sitr = elasticityNeighborsTmpPtr1->find(ElasticityTrackerData(_cell2));
        if (sitr != elasticityNeighborsTmpPtr1->end()) {
            return const_cast<ElasticityTrackerData *>(&(*sitr));
        } else {
            return 0;
        }

    }

}