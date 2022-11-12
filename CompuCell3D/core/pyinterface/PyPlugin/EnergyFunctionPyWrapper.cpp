#include "EnergyFunctionPyWrapper.h"
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <Logger/CC3DLogger.h>
#include <iostream>

using namespace std;
using namespace CompuCell3D;        
EnergyFunctionPyWrapper::EnergyFunctionPyWrapper():
EnergyFunction()
{
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
}
EnergyFunctionPyWrapper::~EnergyFunctionPyWrapper()

{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	

}




EnergyFunction * EnergyFunctionPyWrapper::getEnergyFunctionPyWrapperPtr()
{return this;}

double EnergyFunctionPyWrapper::changeEnergy(const Point3D &pt, const CellG *_newCell,
            const CellG *_oldCell)
{
   int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
   changePointVec[currentWorkNodeNumber]=pt;
   flipNeighborVec[currentWorkNodeNumber]=potts->getFlipNeighbor();

   newCellVec[currentWorkNodeNumber]=const_cast<CellG *>(_newCell);
   oldCellVec[currentWorkNodeNumber]=const_cast<CellG *>(_oldCell);
   

   double energy=0.0;
   PyObject *ret;

   //Using OpenMp lock here - single thread will anly execute this code fragment - gives much better performance than using GIL only 
   //Maybe IronPython could be the solution... will have to explore it - looks like it is much better for multi-core applications.
   pUtils->setLock(lockPtr);

   CC3D_Log(LOG_TRACE) << "currentWorkNodeNumber="<<currentWorkNodeNumber;
	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
   for (int i = 0 ; i < vecPyObject.size() ; ++i){
	     ret=PyObject_CallMethod(vecPyObject[i],"changeEnergy",0);
//       Py_DECREF(ret);
//       energy=PyArg_ParseTuple(ret,"f",&energy);
         energy+=PyFloat_AsDouble(ret);
         Py_DECREF(ret);
         //will need to decrement reference here
         CC3D_Log(LOG_TRACE) << "ENERGY FROM INSIDE WRAPPER=" << energy;
   }

   PyGILState_Release(gstate);

   pUtils->unsetLock(lockPtr);

   return energy;   
//     return 10.0;
}


double EnergyFunctionPyWrapper::localEnergy(const Point3D &pt){return 0.0;}

void EnergyFunctionPyWrapper::registerPyEnergyFunction(PyObject * _enFunction){
   registerPyObject(_enFunction);

//    vecPyObject.push_back(_enFunction);

}