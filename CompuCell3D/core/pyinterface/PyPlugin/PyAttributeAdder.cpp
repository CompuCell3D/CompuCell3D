#include "PyAttributeAdder.h"
#include <CompuCell3D/Potts3D/Cell.h>
#include <Logger/CC3DLogger.h>
#include <Python.h>
#include <iostream>


using namespace CompuCell3D;
using namespace std;

void PyAttributeAdder::addAttribute(CellG * _cell){
	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	
   CC3D_Log(LOG_TRACE) << "Adding new attribute";
   CC3D_Log(LOG_TRACE) << "this is pyAddress="<<_cell->pyAttrib<< " cell id="<<_cell->id;
   PyObject *obj;
   obj = PyObject_CallMethod(vecPyObject[0],"addAttribute",0);
   _cell->pyAttrib=obj;
	PyGILState_Release(gstate);

}

AttributeAdder * PyAttributeAdder::getPyAttributeAdderPtr(){
    return this;
}

void PyAttributeAdder::destroyAttribute(CellG * _cell){

   // since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
   //we aquire GIL and then release it once we are done
   PyGILState_STATE gstate;
   gstate = PyGILState_Ensure();
        
   // PyDict_Clear(_cell->pyAttrib) ; // perhaps we should call this explicitely to clear dictionary before decrementing reference count
   Py_DECREF(_cell->pyAttrib);
   
   PyGILState_Release(gstate);
}


void PyAttributeAdder::registerAdder(PyObject *_adder){
   registerPyObject(_adder);

}