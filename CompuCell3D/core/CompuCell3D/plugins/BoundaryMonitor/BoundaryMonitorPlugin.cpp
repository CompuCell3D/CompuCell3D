
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include "BoundaryMonitorPlugin.h"


BoundaryMonitorPlugin::BoundaryMonitorPlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        boundaryStrategy(0),
        maxNeighborIndex(0) {}

BoundaryMonitorPlugin::~BoundaryMonitorPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;

    if (boundaryArray) {
        delete boundaryArray;
        boundaryArray = 0;
    }

}

Array3DCUDA<unsigned char> *BoundaryMonitorPlugin::getBoundaryArray() { return boundaryArray; }


void BoundaryMonitorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
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
    unsigned char initVal = 0;
    boundaryArray = new Array3DCUDA<unsigned char>(fieldDim, initVal); // 0 indicates pixels is not a boundary pixel

    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);


    potts->registerCellGChangeWatcher(this);


    simulator->registerSteerableObject(this);
}

void BoundaryMonitorPlugin::extraInit(Simulator *simulator) {

}


void BoundaryMonitorPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    if (newCell ==
        oldCell) {//happens during multiple calls to se fcn on the same pixel woth current cell - should be avoided
        return;
    }
    //This function will be called after each successful pixel copy - field3DChange does usual housekeeping
    // tasks to make sure state of cells, and state of the lattice is uptodate

    CellG *nCell;// neighbor cell
    CellG *nnCell; // neighbor of a neighbor cell

    Neighbor neighbor;
    Neighbor nneighbor;

    boundaryArray->set(pt, 0);//assume that pt looses status of boundary pixel
    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }

        nCell = cellFieldG->get(neighbor.pt);
        if (nCell != newCell) { //if newPixel touches cell of different type this means it is a boundary pixel
            boundaryArray->set(pt, 1);
//             break;
        }


        boundaryArray->set(neighbor.pt, 0);//assume that neighbors of pt loose status of boundary pixels
        for (unsigned int nnIdx = 0; nnIdx <= maxNeighborIndex; ++nnIdx) {
            nneighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(neighbor.pt), nnIdx);
            if (!nneighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            nnCell = cellFieldG->get(nneighbor.pt);

            if (nneighbor.pt == pt) {// after pixel copy pt will be occupied by new cell
                if (nCell != newCell) {
                    boundaryArray->set(neighbor.pt, 1);//
                    break;

                }

            } else {
                if (cellFieldG->get(nneighbor.pt) != nCell) {
                    boundaryArray->set(neighbor.pt, 1);
                    break;
                }
            }

        }

    }
}

void BoundaryMonitorPlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id != LATTICE_RESIZE) {
        return;
    }


    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);
    Dim3D newDim = ev.newDim;
    Dim3D oldDim = ev.oldDim;
    Dim3D shiftVec = ev.shiftVec;

    unsigned char initVal = 0;
    Array3DCUDA<unsigned char> *newBoundaryArray = new Array3DCUDA<unsigned char>(newDim,
                                                                                  initVal); // 0 indicates pixels is not a boundary pixel

    Point3D pt;
    Point3D ptShift;

    //when lattice is growing or shrinking
    for (pt.x = 0; pt.x < newDim.x; ++pt.x)
        for (pt.y = 0; pt.y < newDim.y; ++pt.y)
            for (pt.z = 0; pt.z < newDim.z; ++pt.z) {

                ptShift = pt - shiftVec;
                if (ptShift.x >= 0 && ptShift.x < oldDim.x && ptShift.y >= 0 && ptShift.y < oldDim.y &&
                    ptShift.z >= 0 && ptShift.z < oldDim.z) {

                    newBoundaryArray->set(pt, boundaryArray->get(ptShift));
                }
            }
    //reassign boundary lattice
    delete boundaryArray;
    boundaryArray = newBoundaryArray;

}

void BoundaryMonitorPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    automaton = potts->getAutomaton();

    //boundaryStrategy has information about pixel neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();

}

std::string BoundaryMonitorPlugin::toString() {
    return "BoundaryMonitor";
}

std::string BoundaryMonitorPlugin::steerableName() {
    return toString();
}
