#include "StepperPyWrapper.h"
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <iostream>

using namespace std;
using namespace CompuCell3D;        
StepperPyWrapper::StepperPyWrapper():
Stepper()
{
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
}
StepperPyWrapper::~StepperPyWrapper()
{
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
}




Stepper* StepperPyWrapper::getStepperPyWrapperPtr()
{return this;}

void StepperPyWrapper::step()
{
   CC3D_Log(LOG_TRACE) << "STEPPER ";
   
   PyObject *ret;
   //Using OpenMp lock here - single thread will anly execute this code fragment - gives much better performance than using GIL only 
   //Maybe IronPython could be the solution... will have to explore it - looks like it is much better for multi-core applications.
   pUtils->setLock(lockPtr);


	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
   for (int i = 0 ; i < vecPyObject.size() ; ++i){
         CC3D_Log(LOG_TRACE) << "before the call";
      ret=PyObject_CallMethod(vecPyObject[i],"step",0);

      

      //decrement reference here
      Py_DECREF(ret);
      CC3D_Log(LOG_TRACE) << "after the call";
   }
	PyGILState_Release(gstate);

	pUtils->unsetLock(lockPtr);

}




void StepperPyWrapper::registerPyStepper(PyObject * _pyStepper){
    registerPyObject(_pyStepper);

}