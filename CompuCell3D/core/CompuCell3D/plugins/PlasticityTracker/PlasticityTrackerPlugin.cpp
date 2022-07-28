#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <ctime>

using namespace CompuCell3D;


using namespace std;


#include "PlasticityTrackerPlugin.h"


PlasticityTrackerPlugin::PlasticityTrackerPlugin() :
        pUtils(0),
        lockPtr(0),
        cellFieldG(0),
        initialized(false),
        maxNeighborIndex(0),
        boundaryStrategy(0),
        xmlData(0) {}

PlasticityTrackerPlugin::~PlasticityTrackerPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}


void PlasticityTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

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



    ///will register PlasticityTracker here
    ExtraMembersGroupAccessorBase *plasticityTrackerAccessorPtr = &plasticityTrackerAccessor;
    ///************************************************************************************************
    ///REMARK. HAVE TO USE THE SAME CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
    ///************************************************************************************************
    potts->getCellFactoryGroupPtr()->registerClass(plasticityTrackerAccessorPtr);


    fieldDim = cellFieldG->getDim();

    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//1st nearest neighbor

    bool pluginAlreadyRegisteredFlag;
    Plugin *plugin = Simulator::pluginManager.get("CenterOfMass",
                                                  &pluginAlreadyRegisteredFlag); //this will load COM plugin if it is not already loaded
    if (!pluginAlreadyRegisteredFlag)
        plugin->init(simulator);


    potts->registerCellGChangeWatcher(
            this);//register plasticityTracker after CenterOfMass and after VolumeTracker - implicitly called from CenterOfmass
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PlasticityTrackerPlugin::extraInit(Simulator *_simulator) {
    plasticityTypesNames.clear();
    plasticityTypes.clear();
    CC3DXMLElementList includeTypeNamesXMLVec = xmlData->getElements("IncludeType");
    for (int i = 0; i < includeTypeNamesXMLVec.size(); ++i) {
        plasticityTypesNames.insert(includeTypeNamesXMLVec[i]->getText());
    }

    Automaton *automaton = simulator->getPotts()->getAutomaton();
    // Initializing set of plasticitytypes
    for (set<string>::iterator sitr = plasticityTypesNames.begin(); sitr != plasticityTypesNames.end(); ++sitr) {
        plasticityTypes.insert(automaton->getTypeId(*sitr));
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void PlasticityTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,
                                            CellG *oldCell) {

    //do not do any updates until the lattice is fully initialized

    Potts3D *potts = simulator->getPotts();

    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    Neighbor neighbor;
    set<unsigned char>::iterator endSitr = plasticityTypes.end();


    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return;
    if (simulator->getStep() >= 0 && !initialized) {
        pUtils->setLock(lockPtr);
        //we will check  initialized flag inside initializePlasticityNeighborList to make sure that
        // only one thread executes this function - there will be many threads that will enter this section
        initializePlasticityNeighborList();
        pUtils->unsetLock(lockPtr);

    }

    if (initialized) {
        if (oldCell && oldCell->volume == 0 && plasticityTypes.find(oldCell->type) != plasticityTypes.end()) {
            //remove oldCell from neighbor list of old cell neighbors
            set<PlasticityTrackerData>::iterator sitr;
            set <PlasticityTrackerData> *plasticityNeighborsPtr = &plasticityTrackerAccessor.get(
                    oldCell->extraAttribPtr)->plasticityNeighbors;
            set <PlasticityTrackerData> *plasticityNeighborsTmpPtr;

            for (sitr = plasticityNeighborsPtr->begin(); sitr != plasticityNeighborsPtr->end(); ++sitr) {
                //getting set of plasticityNeighbors from the neighbor (pointed by sitr) of the oldCell

                plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(
                        sitr->neighborAddress->extraAttribPtr)->plasticityNeighbors;
//
            }
        }
    }

    std::set <NeighborSurfaceData> *neighborData;
    std::set<NeighborSurfaceData>::iterator sitr;
    CellG *nCell;
    bool pluginAlreadyRegisteredFlag;
    NeighborTrackerPlugin *neighborTrackerPluginPtr = (NeighborTrackerPlugin * )(
            Simulator::pluginManager.get("NeighborTracker", &pluginAlreadyRegisteredFlag));
    neighborTrackerAccessorPtr = neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();

    if (oldCell) {
        neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);

        set<PlasticityTrackerData>::iterator PlasSetitr;
        set<PlasticityTrackerData>::iterator tmpPlasSetitr;
        set <PlasticityTrackerData> *plasticityNeighborsPtr = &plasticityTrackerAccessor.get(
                oldCell->extraAttribPtr)->plasticityNeighbors;
        set <PlasticityTrackerData> OGplasticityNeighborsPtr = plasticityTrackerAccessor.get(
                oldCell->extraAttribPtr)->plasticityNeighbors;
        set <PlasticityTrackerData> *plasticityNeighborsTmpPtr;

        plasticityNeighborsPtr->clear();
        for (sitr = neighborData->begin(); sitr != neighborData->end(); ++sitr) {
            nCell = sitr->neighborAddress;

            if (nCell) {
                if (plasticityTypes.find(nCell->type) == endSitr) {
                } else {
                    plasticityNeighborsPtr->insert(PlasticityTrackerData(nCell));
                    plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(
                            nCell->extraAttribPtr)->plasticityNeighbors;
                    plasticityNeighborsTmpPtr->insert(PlasticityTrackerData(oldCell));
                }
            }
        }
        for (PlasSetitr = plasticityNeighborsPtr->begin(); PlasSetitr != plasticityNeighborsPtr->end(); ++PlasSetitr) {
            plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(oldCell->extraAttribPtr)->plasticityNeighbors;
        }
        for (PlasSetitr = OGplasticityNeighborsPtr.begin();
             PlasSetitr != OGplasticityNeighborsPtr.end(); ++PlasSetitr) {

            plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(
                    PlasSetitr->neighborAddress->extraAttribPtr)->plasticityNeighbors;
            for (tmpPlasSetitr = plasticityNeighborsTmpPtr->begin();
                 tmpPlasSetitr != plasticityNeighborsTmpPtr->end(); ++tmpPlasSetitr) {

            }


        }

    }


}


void PlasticityTrackerPlugin::initializePlasticityNeighborList() {

    //we double check this flag to makes sure this function does not get called multiple times by different threads
    if (initialized)
        return;

    Point3D pt;
    CellG *cell;
    CellG *nCell;
    Neighbor neighbor;
    set <PlasticityTrackerData> *plasticityNeighborsTmpPtr;
    set<unsigned char>::iterator endSitr = plasticityTypes.end();


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
                            cell) {//only cells which are of certain types are considered as plasticityNeighbors
                            if (plasticityTypes.find(nCell->type) != endSitr &&
                                plasticityTypes.find(cell->type) != endSitr) {
                                plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(
                                        nCell->extraAttribPtr)->plasticityNeighbors;
                                plasticityNeighborsTmpPtr->insert(PlasticityTrackerData(cell));
                                plasticityNeighborsTmpPtr = &plasticityTrackerAccessor.get(
                                        cell->extraAttribPtr)->plasticityNeighbors;
                                plasticityNeighborsTmpPtr->insert(PlasticityTrackerData(nCell));
                            }
                        }
                    }
                }
            }
    initialized = true;
}

void PlasticityTrackerPlugin::addPlasticityNeighborList() {}
