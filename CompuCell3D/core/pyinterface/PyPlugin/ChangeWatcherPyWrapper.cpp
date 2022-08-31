#include "ChangeWatcherPyWrapper.h"
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <iostream>

using namespace std;
using namespace CompuCell3D;        
ChangeWatcherPyWrapper::ChangeWatcherPyWrapper():
CellGChangeWatcher()
{
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
}
ChangeWatcherPyWrapper::~ChangeWatcherPyWrapper()
{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
}




CellGChangeWatcher * ChangeWatcherPyWrapper::getChangeWatcherPyWrapperPtr()
{return this;}

void ChangeWatcherPyWrapper::field3DChange(const Point3D &pt, CellG *_newCell,CellG *_oldCell)
{
   
   //return;
   int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	
   //cerr<<"CHANGE WATCHER pt="<<pt<<" _newCell="<<_newCell<<" _oldCell="<<_oldCell<<" currentWorkNodeNumber="<<currentWorkNodeNumber<<" newCellVec.size()="<<newCellVec.size()<<endl;
   changePointVec[currentWorkNodeNumber]=pt;
   //Notice, we cannot be accessing flip neighbor because change watchers are called even before pixel copy begins (e.g. during initialization of cell field)
   //flipNeighborVec[currentWorkNodeNumber]=potts->getFlipNeighbor();

   newCellVec[currentWorkNodeNumber]=const_cast<CellG *>(_newCell);
   oldCellVec[currentWorkNodeNumber]=const_cast<CellG *>(_oldCell);

   
   PyObject *ret;

   //Using OpenMp lock here - single thread will anly execute this code fragment - gives much better performance than using GIL only 
   //Maybe IronPython could be the solution... will have to explore it - looks like it is much better for multi-core applications.

   pUtils->setLock(lockPtr);

	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done



	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
   for (int i = 0 ; i < vecPyObject.size() ; ++i){

      //cerr<<"before the call"<<endl;
      ret=PyObject_CallMethod(vecPyObject[i],"field3DChange",0);
	  //cerr<<"ret="<<ret<<endl; 	
      

      //decrement reference here
      Py_DECREF(ret);

	
//       cerr<<"after the call"<<endl;
   }
	PyGILState_Release(gstate);

	pUtils->unsetLock(lockPtr);
}




void ChangeWatcherPyWrapper::registerPyChangeWatcher(PyObject * _fieldWatcher){
    registerPyObject(_fieldWatcher);

}