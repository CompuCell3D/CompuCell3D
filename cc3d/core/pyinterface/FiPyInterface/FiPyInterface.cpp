#include <iostream>
#include <CompuCell3D/Field3D/Field3D.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "FiPyInterface.h"

using namespace std;
using namespace CompuCell3D;



FiPyInterfaceBase::FiPyInterfaceBase(int _dim) {
//   dimPtr=&FiPyInterfaceBase::secrete;
//   FiPyInterfaceBase::*dimPtr
    if(_dim < 3) {
      dimPtr=&FiPyInterfaceBase::test1;
    }
    else {
      dimPtr= &FiPyInterfaceBase::test2;
    }
}

FiPyInterfaceBase::~FiPyInterfaceBase() {}

void FiPyInterfaceBase::test1() {
  cout << "test1" << endl;
}

void FiPyInterfaceBase::test2() {
  cout << "test2" << endl;
}

void FiPyInterfaceBase::fillArray3D(PyObject * _FiPyArray,CompuCell3D::Field3D<float>* _Field) {
    Dim3D dim=_Field->getDim();
    Point3D pt;
    float sum = 0;
//     cout <<" Dim.x: " << dim.x << " Dim.y: " << dim.y << " Dim.z: " << dim.z << endl;
	for(pt.x =0 ; pt.x < dim.x; ++pt.x) {
		for(pt.y =0 ; pt.y < dim.y; ++pt.y) {
			for(pt.z =0 ; pt.z < dim.z; ++pt.z){
			  sum += _Field->get(pt);
			}
		}
	}
	
	double *dptr;  /* could make this any variable type */
	dptr = (double *)PyArray_DATA(_FiPyArray);
	Py_INCREF(dptr);
// 	(this->*dimPtr)();
	
	int NumElements = PyArray_DIM(_FiPyArray,0);	
	int dims = PyArray_NDIM(_FiPyArray);
	
// 	cout << "Number of Elements: " << NumElements << endl;
// 	cout << "Number of Dimensions: " << dims << endl;
	
	for(int i = 0; i < NumElements; i++) {
	  
	  //3D to 1D
	  /*
	  pt.x = i/(dim.x*dim.y);
	  pt.y = (i-pt.x*dim.x*dim.y)/dim.x;
	  pt.z = i-pt.x*dim.x*dim.y-pt.y*dim.y;
	  */
	  
	//2D to 1D
	  pt.x = i/dim.x;
	  pt.y = (i%dim.y);
	  pt.z = 0;
	  float value = (float)(*dptr);
	  
	  _Field->set(pt,value);
	  dptr++;
	}
	Py_DECREF(dptr);
	dptr = (double *)PyArray_DATA(_FiPyArray);
	Py_INCREF(dptr);
	std::vector<std::vector<float> > secData =  _Field->getSecretionData();
	
	for(int i = 0; i < secData.size(); i++) {
	      dptr[((int)secData[i][0]-1)*dim.x+(int)secData[i][1]-1] += secData[i][3]; //need offset of one for ghost row on concentration?
	}
	_Field->clearSecData();
	Py_DECREF(dptr);
	doNotDiffuseVec = _Field->getDoNotDiffuseVec();
	
// 	cout << "C++ Sum: " << sum << endl;
// 	cout <<" Dim.x: " << dim.x << " Dim.y: " << dim.y << endl;
	

}


void FiPyInterfaceBase::ExampleFunc(PyObject * _FiPyArray,CompuCell3D::Field3D<float>* _Field) {
    Dim3D dim=_Field->getDim();
    Point3D pt;
    float sum = 0;
    cout <<" Dim.x: " << dim.x << " Dim.y: " << dim.y << endl;
	for(pt.x =0 ; pt.x < dim.x; ++pt.x) {
		for(pt.y =0 ; pt.y < dim.y; ++pt.y) {
			for(pt.z =0 ; pt.z < dim.z; ++pt.z){
				sum += _Field->get(pt);
			}
		}
	}
	
	cout << "C++ Sum: " << sum << endl;
	cout <<" Dim.x: " << dim.x << " Dim.y: " << dim.y << endl;
	
}

void FiPyInterfaceBase::doNotDiffuseList(vector<int>& _vec){
  for(int i = 0; i< 100; i++) {
    cout << "i: " << i << endl;
    _vec.push_back(100);
  }
//   std::vector<int>::iterator INTitr;
//   INTitr = doNotDiffuseList.begin();
//   return doNotDiffuseList;
}

std::vector<std::vector<int> > FiPyInterfaceBase::getDoNoDiffuseVec() {
  return doNotDiffuseVec;
}

void FiPyInterfaceBase::test(std::vector<int>& v) {
        for(int i = 0; i < 10; i++) {
                v.push_back(i);
        }
}
