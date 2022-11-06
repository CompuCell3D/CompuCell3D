#include <CompuCell3D/CC3D.h>


using namespace CompuCell3D;
using namespace std;

#include "VolumeTrackerPlugin.h"
#include <Logger/CC3DLogger.h>
VolumeTrackerPlugin::VolumeTrackerPlugin() : pUtils(0), lockPtr(0), potts(0), deadCellG(0) {
}

VolumeTrackerPlugin::~VolumeTrackerPlugin() {
    pUtils->destroyLock(lockPtr);
    delete lockPtr;
    lockPtr = 0;
}

void VolumeTrackerPlugin::initVec(const vector<int> &_vec) {
    CC3D_Log(LOG_DEBUG) << " THIS IS VEC.size="<<_vec.size();
}

void VolumeTrackerPlugin::initVec(const Dim3D &_dim) {
    CC3D_Log(LOG_DEBUG) << " THIS IS A COMPUCELL3D DIM3D"<<_dim;
}

bool VolumeTrackerPlugin::checkIfOKToResize(Dim3D _newSize, Dim3D _shiftVec) {

    Field3DImpl < CellG * > *cellField = (Field3DImpl < CellG * > *)
    potts->getCellFieldG();
    Dim3D fieldDim = cellField->getDim();
    Point3D pt;
    Point3D shiftVec(_shiftVec.x, _shiftVec.y, _shiftVec.z);
    Point3D shiftedPt;
    CellG *cell;

    for (pt.x=0 ; pt.x<fieldDim.x ; ++pt.x)
		for (pt.y=0 ; pt.y<fieldDim.y ; ++pt.y)
			for (pt.z=0 ; pt.z<fieldDim.z ; ++pt.z){
				cell=cellField->get(pt);
				if(cell){
					shiftedPt=pt+shiftVec;

                    if (shiftedPt.x < 0 || shiftedPt.x >= _newSize.x || shiftedPt.y < 0 || shiftedPt.y >= _newSize.y ||
                        shiftedPt.z < 0 || shiftedPt.z >= _newSize.z) {
                        return false;
                    }
                }

            }
    return true;
}


void VolumeTrackerPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
    sim = simulator;
    pUtils = sim->getParallelUtils();
    lockPtr = new ParallelUtilsOpenMP::OpenMPLock_t;
    pUtils->initLock(lockPtr);


    potts = simulator->getPotts();
    potts->registerCellGChangeWatcher(this);
    potts->registerStepper(this);

    deadCellVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), (CellG *) 0);
}


void VolumeTrackerPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){
		CC3DEventChangeNumberOfWorkNodes ev = static_cast<CC3DEventChangeNumberOfWorkNodes&>(_event);
		deadCellVec.assign(pUtils->getMaxNumberOfWorkNodesPotts(), (CellG*)0);
		CC3D_Log(LOG_DEBUG) << "VolumeTrackerPlugin::handleEvent=";
    }
}


std::string VolumeTrackerPlugin::toString() { return "VolumeTracker"; }

std::string VolumeTrackerPlugin::steerableName() { return toString(); }


void VolumeTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

    if (newCell)
        newCell->volume++;

    if (oldCell)

        if ((--oldCell->volume) == 0) {

            deadCellVec[pUtils->getCurrentWorkNodeNumber()] = oldCell;
        }


}

//have to fix it
void VolumeTrackerPlugin::step() {

    CellG *deadCellPtr = deadCellVec[pUtils->getCurrentWorkNodeNumber()];
    if (deadCellPtr) {

        //NOTICE: we cannot use #pragma omp critical instead of
        // locks because although this is called from inside the parallel region critical directive has to be included explicitely inside #pragma omp parallel section - and this has to be known at the compile time
        pUtils->setLock(lockPtr);

        potts->destroyCellG(deadCellPtr);
        deadCellVec[pUtils->getCurrentWorkNodeNumber()] = 0;
        pUtils->unsetLock(lockPtr);
    }


}

