
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include "CurvatureCalculatorPlugin.h"


CurvatureCalculatorPlugin::CurvatureCalculatorPlugin() :
        pUtils(0),
        lockPtr(0),
        xmlData(0),
        cellFieldG(0),
        boundaryStrategy(0),
        maxNeighborIndex(0),
        neighborTrackerAccessorPtr(0),
        boundary_pixel_tracker_plugin(0),
        neighborOrderProbCalc(4) {}

CurvatureCalculatorPlugin::~CurvatureCalculatorPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void CurvatureCalculatorPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    xmlData = _xmlData;
    sim = simulator;
    potts = simulator->getPotts();
    cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    pUtils = sim->getParallelUtils();

    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);

    boundaryStrategy = BoundaryStrategy::getInstance();
    maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

    potts->registerCellGChangeWatcher(this);

    simulator->registerSteerableObject(this);
}


void CurvatureCalculatorPlugin::extraInit(Simulator *simulator) {

    update(xmlData, true);

    maxNeighborIndexProbCalc = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrderProbCalc);

    bool neighborTrackerAlreadyRegisteredFlag;
    Plugin *nt_plugin = Simulator::pluginManager.get("NeighborTracker",
                                                     &neighborTrackerAlreadyRegisteredFlag);  //this will load NeighborTracker plugin if it is not already loaded
    NeighborTrackerPlugin *neighborTrackerPlugin = (NeighborTrackerPlugin *) nt_plugin;
    if (!neighborTrackerAlreadyRegisteredFlag) {
        neighborTrackerPlugin->init(simulator);
    }

    neighborTrackerAccessorPtr = neighborTrackerPlugin->getNeighborTrackerAccessorPtr();

    bool boundaryPixelTrackerAlreadyRegisteredFlag;

    boundary_pixel_tracker_plugin = (BoundaryPixelTrackerPlugin *) Simulator::pluginManager.get("BoundaryPixelTracker",
                                                                                                &boundaryPixelTrackerAlreadyRegisteredFlag);
    if (!boundaryPixelTrackerAlreadyRegisteredFlag) {
        boundary_pixel_tracker_plugin->init(simulator);
    }
}


void CurvatureCalculatorPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    //This function will be called after each successful pixel copy - field3DChange does usual housekeeping
    // tasks to make sure state of cells, and state of the lattice is up-to-date
    if (newCell) {
        //PUT YOUR CODE HERE
    } else {
        //PUT YOUR CODE HERE
    }

    if (oldCell) {
        //PUT YOUR CODE HERE
    } else {
        //PUT YOUR CODE HERE
    }

}

std::map<long, float> CurvatureCalculatorPlugin::getProbabilityByNeighbor(CellG *cell, float J, float T) {

    map<long, float> id_to_prob_map;

    if (!cell) {
        return id_to_prob_map;
    }

    NeighborTracker *neighborTracker = neighborTrackerAccessorPtr->get(cell->extraAttribPtr);

    set<NeighborSurfaceData>::iterator neighborItr;

    for (neighborItr = neighborTracker->cellNeighbors.begin();
         neighborItr != neighborTracker->cellNeighbors.end(); ++neighborItr) {
        if (neighborItr->neighborAddress) {
            id_to_prob_map.insert(std::make_pair(neighborItr->neighborAddress->id, 0.0));
        }
    }

    if (!id_to_prob_map.size())
        return id_to_prob_map;

    Neighbor neighbor;
    CellG *nCell;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    ExtraMembersGroupAccessor <BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr = boundary_pixel_tracker_plugin->getBoundaryPixelTrackerAccessorPtr();
    std::set <BoundaryPixelTrackerData> &pixelSetRef = boundaryPixelTrackerAccessorPtr->get(
            cell->extraAttribPtr)->pixelSet;


    for (set<BoundaryPixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {
        Point3D pt = sitr->pixel;

        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            if (!neighbor.distance) {
                //if distance is 0 then the neighbor returned is invalid
                continue;
            }

            nCell = fieldG->get(neighbor.pt);

            if (nCell != cell) {
                float prob = getGrowthProbability(neighbor.pt, nCell, cell, J, T);
                id_to_prob_map[nCell->id] += prob;
            }
        }

    }

    return id_to_prob_map;

}

float
CurvatureCalculatorPlugin::getGrowthProbability(const Point3D &neighborPt, CellG *neighborCell, CellG *cell, float J,
                                                float T) {
    double E0 = 0.0;
    double Ef = 0.0;

    Neighbor neighbor;
    CellG *local_neighbor_cell;
    WatchableField3D < CellG * > *fieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();

    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndexProbCalc; ++nIdx) {
        neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(neighborPt), nIdx);
        if (!neighbor.distance) {
            //if distance is 0 then the neighbor returned is invalid
            continue;
        }

        local_neighbor_cell = fieldG->get(neighbor.pt);
        // current energy (when pixel still belongs to NCELL)
        if (local_neighbor_cell != neighborCell) {
            E0 += J;
        }

        // flipped energy (when pixel now belongs to CELL)
        if (local_neighbor_cell != cell) {
            Ef += J;
        }

    }

    if (Ef <= E0) {
        return 1.0;
    } else {

        return exp(-(Ef - E0) / T);
    }

}

void CurvatureCalculatorPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {
    //PARSE XML IN THIS FUNCTION
    //For more information on XML parser function please see CC3D code or lookup XML utils API
    automaton = potts->getAutomaton();
    if (!automaton)
        throw CC3DException(
                "CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET");
    set<unsigned char> cellTypesSet;

    CC3DXMLElement *jXMLElem = _xmlData->getFirstElement("J");

    CC3DXMLElement *tXMLElem = _xmlData->getFirstElement("T");

    CC3DXMLElement *neighborOrderProbCalcXMLElem = _xmlData->getFirstElement("NeighborOrderProbCalc");

    if (neighborOrderProbCalcXMLElem) {
        neighborOrderProbCalc = neighborOrderProbCalcXMLElem->getInt();
    }

    //boundaryStrategy has information aobut pixel neighbors
    boundaryStrategy = BoundaryStrategy::getInstance();

}


std::string CurvatureCalculatorPlugin::toString() {
    return "CurvatureCalculator";
}


std::string CurvatureCalculatorPlugin::steerableName() {
    return toString();
}
