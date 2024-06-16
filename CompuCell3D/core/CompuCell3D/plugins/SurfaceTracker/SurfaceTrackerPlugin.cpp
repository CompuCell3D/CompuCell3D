#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;


using namespace std;


#include "SurfaceTrackerPlugin.h"

SurfaceTrackerPlugin::SurfaceTrackerPlugin() : cellFieldG(0), boundaryStrategy(0), maxNeighborIndex(0) {}

SurfaceTrackerPlugin::~SurfaceTrackerPlugin() {
}


void SurfaceTrackerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    potts->registerCellGChangeWatcher(this);

    boundaryStrategy = BoundaryStrategy::getInstance();

    update(_xmlData);

    simulator->registerSteerableObject(this);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SurfaceTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return;

    unsigned int token = 0;
    double distance;
    double oldDiff = 0.;
    double newDiff = 0.;
    CellG *nCell = 0;
    Neighbor neighbor;

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
        if (!neighbor.distance)
            continue;

        nCell = cellFieldG->get(neighbor.pt);
        if (newCell == nCell) newDiff -= lmf.surfaceMF;
        else newDiff += lmf.surfaceMF;

        if (oldCell == nCell) oldDiff += lmf.surfaceMF;
        else oldDiff -= lmf.surfaceMF;
    }

    if (newCell) newCell->surface += newDiff;
    if (oldCell) oldCell->surface += oldDiff;
}


void SurfaceTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    if (!_xmlData) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                1); //use first nearest neighbors for surface calculations as default
    } else if (_xmlData->getFirstElement("NeighborOrder")) {
        setNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());
    }
    else if (_xmlData->getFirstElement("MaxNeighborOrder")) {
        setNeighborOrder(_xmlData->getFirstElement("MaxNeighborOrder")->getUInt());
    } else if (_xmlData->getFirstElement("MaxNeighborDistance")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(
                _xmlData->getFirstElement("MaxNeighborDistance")->getDouble());//depth=1.1 - means 1st nearest neighbor
    } else {
        setNeighborOrder(1);
    }
    lmf = boundaryStrategy->getLatticeMultiplicativeFactors();
}

void SurfaceTrackerPlugin::setNeighborOrder(unsigned int neighbor_order){
    ASSERT_OR_THROW("boundaryStrategy not initializer", boundaryStrategy);

    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighbor_order);

}

std::string SurfaceTrackerPlugin::steerableName() { return toString(); }

std::string SurfaceTrackerPlugin::toString() { return "SurfaceTracker"; }