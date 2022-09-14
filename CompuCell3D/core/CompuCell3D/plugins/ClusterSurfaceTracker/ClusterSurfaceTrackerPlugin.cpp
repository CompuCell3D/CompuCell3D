#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;


#include "ClusterSurfaceTrackerPlugin.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h"
#include "CompuCell3D/plugins/PixelTracker/PixelTracker.h"


ClusterSurfaceTrackerPlugin::ClusterSurfaceTrackerPlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        maxNeighborIndex(0) {}

ClusterSurfaceTrackerPlugin::~ClusterSurfaceTrackerPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void ClusterSurfaceTrackerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    bool pluginAlreadyRegisteredFlag;
    //this will load VolumeTracker plugin if it is not already loaded
    pixelTrackerPlugin = (PixelTrackerPlugin *) Simulator::pluginManager.get("PixelTracker",
                                                                             &pluginAlreadyRegisteredFlag);
    if (!pluginAlreadyRegisteredFlag)
        pixelTrackerPlugin->init(simulator);

    pixelTrackerAccessorPtr = pixelTrackerPlugin->getPixelTrackerAccessorPtr();


    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    update(xmlData, true);


    potts->registerCellGChangeWatcher(this);

}

// ExtraInit functions are called in the order plugin was fisrt loaded to set of active plugins inside pluginManager
// Notice that this order is in general not the same as the order of two trackers then tracker 2 inserts tracker 1
// before itself in the list of trackers
// If tracker 2 is first accessed before tracker 1 then order pluginManager order is tracker 2,
// tracker 1 whereas tracker registry has them ordered tracker 1 , tracker 2
// the bottom line is that we can rely on ordering in tracker registry, energy registry etc.
// but cannot rely on ordering when it comes to extraInit fcns
// All the order dependent action should be preferable be done in the init fcns.
// and of course init is always called before extra init so this gives extra wiggle room
void ClusterSurfaceTrackerPlugin::extraInit(Simulator *simulator) {
}


void ClusterSurfaceTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

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
   CC3D_Log(LOG_TRACE) << "NEW EVENT";
    if (newCell) {
        CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(newCell->clusterId);
        //first make sure all compartments have same cluster surface - important during addition of new cluster e.g. during initialization
        double clusterSurface;
        for (int i = 0; i < compartments.size(); ++i) {
            if (i == 0) {
                clusterSurface = compartments[i]->clusterSurface;
            }
            compartments[i]->clusterSurface = clusterSurface;
        }
        //assigning new cluster surface to all members of a cluster

        for (int i = 0; i < compartments.size(); ++i) {

            compartments[i]->clusterSurface += newDiff;
        }
    }

    if (oldCell) {
        //assigning new cluster surface to all members of a cluster
        CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(oldCell->clusterId);

        double clusterSurface;
        for (int i = 0; i < compartments.size(); ++i) {
            if (i == 0) {
                clusterSurface = compartments[i]->clusterSurface;
            }
            compartments[i]->clusterSurface = clusterSurface;
        }

        for (int i = 0; i < compartments.size(); ++i) {

            compartments[i]->clusterSurface += oldDiff;
        }
    }

}

void ClusterSurfaceTrackerPlugin::updateClusterSurface(long _clusterId) {
    CC3DCellList compartments = potts->getCellInventory().getClusterInventory().getClusterCells(_clusterId);
    double clusterSurface = 0.0;

    CellG *nCell;
    for (int i = 0; i < compartments.size(); ++i) {
        CellG *cell = compartments[i];

        set <PixelTrackerData> &cellPixels = pixelTrackerAccessorPtr->get(cell->extraAttribPtr)->pixelSet;
        for (set<PixelTrackerData>::iterator sitr = cellPixels.begin(); sitr != cellPixels.end(); ++sitr) {


            Neighbor neighbor;
            for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
                neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(sitr->pixel), nIdx);
                if (!neighbor.distance) {
                    //if distance is 0 then the neighbor returned is invalid
                    continue;
                }
                nCell = cellFieldG->get(neighbor.pt);
                if (!nCell || (nCell && nCell->clusterId != cell->clusterId))
                    clusterSurface += lmf.surfaceMF;

            }

        }
    }

    for (int i = 0; i < compartments.size(); ++i) {
        compartments[i]->clusterSurface = clusterSurface;
    }

}


void ClusterSurfaceTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    //boundaryStrategy has information aobut pixel neighbors 
    boundaryStrategy = BoundaryStrategy::getInstance();

    if (!_xmlData) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                1); //use first nearest neighbors for surface calculations as default
    } else if (_xmlData->getFirstElement("MaxNeighborOrder")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                _xmlData->getFirstElement("MaxNeighborOrder")->getUInt());
    } else if (_xmlData->getFirstElement("MaxNeighborDistance")) {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromDepth(
                _xmlData->getFirstElement("MaxNeighborDistance")->getDouble());//depth=1.1 - means 1st nearest neighbor
    } else {
        maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(
                1); //use first nearest neighbors for surface calculations as default
    }
    lmf = boundaryStrategy->getLatticeMultiplicativeFactors();


}


std::string ClusterSurfaceTrackerPlugin::toString() {
    return "ClusterSurfaceTracker";
}


std::string ClusterSurfaceTrackerPlugin::steerableName() {
    return toString();
}
