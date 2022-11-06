#include "TypeChangeWatcherPyWrapper.h"
#include <Logger/CC3DLogger.h>
#include <iostream>

using namespace std;
using namespace CompuCell3D;        
TypeChangeWatcherPyWrapper::TypeChangeWatcherPyWrapper():
TypeChangeWatcher()
{
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
}
TypeChangeWatcherPyWrapper::~TypeChangeWatcherPyWrapper()
{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
}




TypeChangeWatcher * TypeChangeWatcherPyWrapper::getTypeChangeWatcherPyWrapperPtr()
{return this;}

void TypeChangeWatcherPyWrapper::typeChange(CellG* _cell,CellG::CellType_t _newType)
{

  int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();
   newCellVec[currentWorkNodeNumber]=const_cast<CellG *>(_cell);
   newTypeVec[currentWorkNodeNumber]=_newType;
   PyObject *ret;

   //Using OpenMp lock here - single thread will anly execute this code fragment - gives much better performance than using GIL only 
   //Maybe IronPython could be the solution... will have to explore it - looks like it is much better for multi-core applications.
   pUtils->setLock(lockPtr);

	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
   for (int i = 0 ; i < vecPyObject.size() ; ++i){
         CC3D_Log(LOG_TRACE) <<  "before the call";
      ret=PyObject_CallMethod(vecPyObject[i],"typeChange",0);

      

      //decrement reference here
      Py_DECREF(ret);
      CC3D_Log(LOG_TRACE) << "after the call";
   }
	PyGILState_Release(gstate);

	pUtils->unsetLock(lockPtr);
}




void TypeChangeWatcherPyWrapper::registerPyTypeChangeWatcher(PyObject * _typeChangeWatcher){
    registerPyObject(_typeChangeWatcher);

}