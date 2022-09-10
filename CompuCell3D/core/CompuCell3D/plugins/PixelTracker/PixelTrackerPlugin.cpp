

#include <string>
#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

using namespace std;

#include "PixelTrackerPlugin.h"


PixelTrackerPlugin::PixelTrackerPlugin() :
        simulator(0), potts(0), pUtils(0) {
    trackMedium = false;
    fullInitAtStart = false;
    fullInitState = false;
}

PixelTrackerPlugin::~PixelTrackerPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}


void PixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


    simulator = _simulator;
    potts = simulator->getPotts();
    pUtils = simulator->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);



    ///will register PixelTracker here
    ExtraMembersGroupAccessorBase *cellPixelTrackerAccessorPtr = &pixelTrackerAccessor;
    ///************************************************************************************************
    ///REMARK. HAVE TO USE THE SAME CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
    ///************************************************************************************************
    potts->getCellFactoryGroupPtr()->registerClass(cellPixelTrackerAccessorPtr);

    potts->registerCellGChangeWatcher(this);

    if (_xmlData) {
        trackMedium = _xmlData->findElement("TrackMedium");
        fullInitAtStart = _xmlData->findElement("FullInitAtStart");
    }

}

void PixelTrackerPlugin::extraInit(Simulator *simulator) {
    if (trackMedium) {

        mediumTrackerDataInit();

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
    if (newCell == oldCell) //this may happen if you are trying to assign same cell to one pixel twice
        return;

    if(fullInitAtStart) {
        fullTrackerDataInit(pt, oldCell);
        fullInitAtStart = false;
    }

    if (newCell) {
        std::set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;
        std::set<PixelTrackerData>::iterator sitr = pixelSetRef.find(PixelTrackerData(pt));
        pixelSetRef.insert(PixelTrackerData(pt));
    } else if (trackMedium) {
        unsigned int workNodeNum = pUtils->getCurrentWorkNodeNumber();
        unsigned int partitionNum = getParitionNumber(pt, workNodeNum);
        mediumPixelSet[partitionNum].insert(PixelTrackerData(pt));
    }

    std::set<PixelTrackerData>::iterator sitr;
    if (oldCell) {
        std::set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
        sitr = pixelSetRef.find(PixelTrackerData(pt));

        if (sitr == pixelSetRef.end())
            throw CC3DException(
                    "Could not find point:" + pt + " inside cell of id: " + std::to_string(oldCell->id) + " type: " +
                    std::to_string((int) oldCell->type));

        pixelSetRef.erase(sitr);
    } else if (trackMedium) {
        unsigned int workNodeNum = pUtils->getCurrentWorkNodeNumber();
        unsigned int partitionNum = getParitionNumber(pt, workNodeNum);
        sitr = mediumPixelSet[partitionNum].find(PixelTrackerData(pt));

        if (sitr == mediumPixelSet[partitionNum].end())
            throw CC3DException("Could not find point:" + pt + " in medium");

        mediumPixelSet[partitionNum].erase(sitr);
    }


}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PixelTrackerPlugin::handleEvent(CC3DEvent &_event) {
    if (_event.id == CHANGE_NUMBER_OF_WORK_NODES) {

        mediumTrackerDataInit();

    } else if (_event.id != LATTICE_RESIZE) {
        return;
    }

    CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize &>(_event);

    Dim3D shiftVec = ev.shiftVec;

    CellInventory &cellInventory = potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG *cell;

    for (cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr) {
        cell = cInvItr->second;
        std::set <PixelTrackerData> &pixelSetRef = pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet;
        for (set<PixelTrackerData>::iterator sitr = pixelSetRef.begin(); sitr != pixelSetRef.end(); ++sitr) {
            Point3D &pixel = const_cast<Point3D &>(sitr->pixel);
            pixel.x += shiftVec.x;
            pixel.y += shiftVec.y;
            pixel.z += shiftVec.z;


        }


    }

    if (trackMedium) {
        for (unsigned int p = 0; p < mediumPixelSet.size(); ++p) {
            for (set<PixelTrackerData>::iterator sitr = mediumPixelSet[p].begin();
                 sitr != mediumPixelSet[p].end(); ++sitr) {
                Point3D &pixel = const_cast<Point3D &>(sitr->pixel);
                pixel.x += shiftVec.x;
                pixel.y += shiftVec.y;
                pixel.z += shiftVec.z;
            }
        }
    }

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string PixelTrackerPlugin::toString() {
    return "PixelTracker";
}

void PixelTrackerPlugin::fullTrackerDataInit(Point3D ptChange, CellG *oldCell) {

	pUtils->setLock(lockPtr);

	CellInventory &cellInventory = potts->getCellInventory();
	CellG *cell;
	for (CellInventory::cellInventoryIterator cInvItr = cellInventory.cellInventoryBegin(); cInvItr != cellInventory.cellInventoryEnd(); ++cInvItr) {
		cell = cInvItr->second;
		pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet.clear();
	}

	WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *>*) potts->getCellFieldG();
	Dim3D fieldDim = cellFieldG->getDim();

	for (auto &mp : mediumPixelSet) 
		mp.clear();

	for (short z = 0; z < fieldDim.z; ++z)
		for (short y = 0; y < fieldDim.y; ++y)
			for (short x = 0; x < fieldDim.x; ++x) {
				Point3D pt = Point3D(x, y, z);
				if (pt == ptChange) { cell = oldCell; } else { cell = cellFieldG->get(pt); }
				if (cell) { pixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet.insert(PixelTrackerData(pt)); }
				else if (trackMedium) {
					mediumPixelSet[getParitionNumber(pt)].insert(PixelTrackerData(pt));
				}
			}

	fullInitState = true;

	pUtils->unsetLock(lockPtr);

}

void PixelTrackerPlugin::mediumTrackerDataInit() {

    pUtils->setLock(lockPtr);

    Field3DImpl < CellG * > *cellFieldG = (Field3DImpl < CellG * > *)
    potts->getCellFieldG();
    Dim3D fieldDim = cellFieldG->getDim();

    unsigned int numSubSecs = pUtils->getNumberOfSubgridSectionsPotts();
    unsigned int numWorkers = pUtils->getNumberOfWorkNodesPotts();

    std::vector <pair<Dim3D, Dim3D>> sectionDimsVecShared;
    for (unsigned int n = 0; n < numWorkers; ++n)
        for (unsigned int s = 0; s < numSubSecs; ++s) {
            pair <Dim3D, Dim3D> pottsSection = pUtils->getPottsSection(n, s);
            if (pottsSection.first.x < pottsSection.second.x ||
                pottsSection.first.y < pottsSection.second.y ||
                pottsSection.first.z < pottsSection.second.z) {
                sectionDimsVecShared.push_back(pottsSection);
            }
        }
    sectionDimsVec.clear();
    sectionDimsVec.assign(numWorkers, sectionDimsVecShared);

    mediumPixelSet.clear();
    mediumPixelSet.assign(sectionDimsVecShared.size(), std::set<PixelTrackerData>());

    for (short z = 0; z < fieldDim.z; ++z)
        for (short y = 0; y < fieldDim.y; ++y)
            for (short x = 0; x < fieldDim.x; ++x) {
                Point3D pt = Point3D(x, y, z);
                CellG *cell = cellFieldG->get(pt);
                if (cell == 0) {
                    unsigned int partitionNumber = getParitionNumber(pt);
                    mediumPixelSet[partitionNumber].insert(PixelTrackerData(pt));
                }
            }

    pUtils->unsetLock(lockPtr);

}

unsigned int PixelTrackerPlugin::getParitionNumber(const Point3D &_pt, unsigned int _workerNum) {

    std::vector <pair<Dim3D, Dim3D>> workerSectionDimsVec = sectionDimsVec[_workerNum];
    for (unsigned int p = 0; p < workerSectionDimsVec.size(); ++p) {
        Dim3D fieldDimMin = workerSectionDimsVec[p].first;
        Dim3D fieldDimMax = workerSectionDimsVec[p].second;
        if ((_pt.x >= fieldDimMin.x && _pt.y >= fieldDimMin.y && _pt.z >= fieldDimMin.z) &&
            (_pt.x < fieldDimMax.x && _pt.y < fieldDimMax.y && _pt.z < fieldDimMax.z)) {
            return p;
        }
    }

    throw CC3DException("Could not find partition for point:" + _pt);
}

// Not thread-safe
std::set <PixelTrackerData> PixelTrackerPlugin::getMediumPixelSet() {

    std::set <PixelTrackerData> mediumPixelSetCombined;
    for (unsigned int p = 0; p < mediumPixelSet.size(); ++p) {
//		for each (PixelTrackerData ptd in mediumPixelSet[p])
        for (auto ptd: mediumPixelSet[p]) {
            mediumPixelSetCombined.insert(ptd);
        }
    }

    return mediumPixelSetCombined;
}
