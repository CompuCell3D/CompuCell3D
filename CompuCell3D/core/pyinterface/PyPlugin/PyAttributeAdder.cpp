#include "PyAttributeAdder.h"
#include <CompuCell3D/Potts3D/Cell.h>
#include <Python.h>
#include <iostream>


using namespace CompuCell3D;
using namespace std;

void PyAttributeAdder::addAttribute(CellG * _cell){
	// since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
	//we aquire GIL and then release it once we are done
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();
	

   //cerr<<"Adding new attribute"<<endl;
   //cerr<<"this is pyAddress="<<_cell->pyAttrib<< " cell id="<<_cell->id<<endl;
   PyObject *obj;
   obj = PyObject_CallMethod(vecPyObject[0],"addAttribute",0);
//    Py_INCREF(obj);
//    Py_INCREF(obj);
   _cell->pyAttrib=obj;

//    cerr<<"refChecker="<<refChecker<<endl;
//    if(refChecker){
//       long refCount;
//       PyObject *refCountPtr;
//       refCountPtr=PyObject_CallObject(refChecker,obj);
//       refCount=PyInt_AsLong(refCountPtr);
//       cerr<<"refrence Count="<<refCount<<endl;
// //       Py_DECREF(refCountPtr);
//    }
//     _cell->pyAttrib=PyList_New(1);
   //cerr<<" after adding extra attrib pyAddress="<<_cell->pyAttrib<<endl;
	PyGILState_Release(gstate);

	

}

AttributeAdder * PyAttributeAdder::getPyAttributeAdderPtr(){
    return this;
}

void PyAttributeAdder::destroyAttribute(CellG * _cell){
   //for debugging purposes you may uncomment this section
   //but make sure that you register with this class getrefcount function from Python level
//     cerr<<"Destroying attribute"<<endl;
//    if(refChecker){
//       long refCount;
//       PyObject *refCountPtr;
//       PyObject *args_tupple=PyTuple_New(1);
//       PyTuple_SetItem(args_tupple,0,_cell->pyAttrib);
//       refCountPtr=PyObject_CallObject(refChecker,args_tupple);
// 
//       refCount=PyInt_AsLong(refCountPtr);
//       cerr<<"before destroying attribute refrence Count="<<refCount<<endl;
//       Py_DECREF(args_tupple);
//       Py_DECREF(refCountPtr);
//    }

        // since we are using threads (swig generated modules are fully "threaded") and and use C/API we better make sure that before doing anything in python
        //we aquire GIL and then release it once we are done
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        
   // // // PyDict_Clear(_cell->pyAttrib) ; // perhaps we should call this explicitely to clear dictionary before decrementing reference count
   Py_DECREF(_cell->pyAttrib);
   
   PyGILState_Release(gstate);
//    Py_DECREF(_cell->pyAttrib);

//    if(destroyer){
//       PyObject *args=PyTuple_New(1);
//       PyTuple_SetItem(args,0,_cell->pyAttrib);
// 
//       PyObject_CallObject(destroyer,args);
//       Py_DECREF(args);
//    }
//    Py_DECREF(_cell->pyAttrib);
//    Py_DECREF(_cell->pyAttrib);
//    Py_DECREF(_cell->pyAttrib);
}


void PyAttributeAdder::registerAdder(PyObject *_adder){
   registerPyObject(_adder);

}