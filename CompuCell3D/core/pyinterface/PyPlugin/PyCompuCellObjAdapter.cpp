#include "PyCompuCellObjAdapter.h"
#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;
using namespace std;

PyCompuCellObjAdapter::PyCompuCellObjAdapter():pUtils(0)
{	
}

void PyCompuCellObjAdapter::setPotts(Potts3D * _potts){
	potts=_potts;
}
void PyCompuCellObjAdapter::setSimulator(Simulator * _sim){
	sim=_sim;
	pUtils=sim->getParallelUtils();
	int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
	newCellVec.assign(maxNumberOfWorkNodes,(CellG*)0);
    oldCellVec.assign(maxNumberOfWorkNodes,(CellG*)0);
    flipNeighborVec.assign(maxNumberOfWorkNodes,Point3D());
    changePointVec.assign(maxNumberOfWorkNodes,Point3D());

}

 bool PyCompuCellObjAdapter::isNewCellValid(){return newCellVec[pUtils->getCurrentWorkNodeNumber()];}
 bool PyCompuCellObjAdapter::isOldCellValid(){return oldCellVec[pUtils->getCurrentWorkNodeNumber()];}
 bool PyCompuCellObjAdapter::isCellMedium(CellG * cell){return !cell;}
 CellG * PyCompuCellObjAdapter::getNewCell(){return newCellVec[pUtils->getCurrentWorkNodeNumber()];}
 CellG * PyCompuCellObjAdapter::getOldCell(){return oldCellVec[pUtils->getCurrentWorkNodeNumber()];}
 Point3D  PyCompuCellObjAdapter::getFlipNeighbor(){return flipNeighborVec[pUtils->getCurrentWorkNodeNumber()];}
 Point3D  PyCompuCellObjAdapter::getChangePoint(){return changePointVec[pUtils->getCurrentWorkNodeNumber()];}
 CellG::CellType_t PyCompuCellObjAdapter::getNewType(){return newTypeVec[pUtils->getCurrentWorkNodeNumber()];}

void PyCompuCellObjAdapter::registerPyObject(PyObject * _pyObject){

   vecPyObject.push_back(_pyObject);
}