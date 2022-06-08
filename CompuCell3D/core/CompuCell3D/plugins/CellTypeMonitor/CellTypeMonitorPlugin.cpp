#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "CellTypeMonitorPlugin.h"


CellTypeMonitorPlugin::CellTypeMonitorPlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        boundaryStrategy(0),
        cellTypeArray(0),
        mediumType(0) {}

CellTypeMonitorPlugin::~CellTypeMonitorPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;

    if (cellTypeArray) {
        delete cellTypeArray;
        cellTypeArray = 0;
        delete cellIdArray;
        cellIdArray = 0;


    }
}

void CellTypeMonitorPlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id != LATTICE_RESIZE) {
        return;
    }

    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);


    Array3DCUDA<unsigned char> *cellTypeArray_new = new Array3DCUDA<unsigned char>(ev.newDim, mediumType);
    //we assume medium cell id is -1 not zero because normally cells in older versions of CC3D
    // we allowed cells with id 0
    Array3DCUDA<float> *cellIdArray_new = new Array3DCUDA<float>(ev.newDim, -1);

    for (int x = 0; x < ev.newDim.x; x++)
        for (int y = 0; y < ev.newDim.y; y++)
            for (int z = 0; z < ev.newDim.z; z++) {
                Point3D pt(x, y, z);
                CellG *cell = cellFieldG->get(pt);
                if (cell) {
                    cellTypeArray_new->set(pt, cell->type);
                    cellIdArray_new->set(pt, cell->id);
                }
            }

    delete cellTypeArray;
    cellTypeArray = 0;
    delete cellIdArray;
    cellIdArray = 0;

    cellTypeArray = cellTypeArray_new;
    cellIdArray = cellIdArray_new;


}


void CellTypeMonitorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    update(xmlData, true);

    Dim3D fieldDim = cellFieldG->getDim();
    cellTypeArray = new Array3DCUDA<unsigned char>(fieldDim, mediumType);
    //we assume medium cell id is -1 not zero because normally cells in older versions of CC3D
    // we allowed cells with id 0
    cellIdArray = new Array3DCUDA<float>(fieldDim, -1);


    potts->registerCellGChangeWatcher(this);


    simulator->registerSteerableObject(this);
}

void CellTypeMonitorPlugin::extraInit(Simulator *simulator) {

}


void CellTypeMonitorPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    //This function will be called after each successful pixel copy - field3DChange does usual housekeeping
    // tasks to make sure state of cells, and state of the lattice is up-to-date
    // here we keep track of a cell type at a given position 
    if (newCell) {
        cellTypeArray->set(pt, newCell->type);
        cellIdArray->set(pt, newCell->id);
    } else {
        cellTypeArray->set(pt, 0);
        cellIdArray->set(pt, 0);
    }

}


void CellTypeMonitorPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");


    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();

}


std::string CellTypeMonitorPlugin::toString() {
    return "CellTypeMonitor";
}


std::string CellTypeMonitorPlugin::steerableName() {
    return toString();
}
