#ifndef FIPYINTERFACE_H
#define FIPYINTERFACE_H

// #include <CompuCell3D/Field3D/Field3D.h>
// #include <CompuCell3D/Field3D/Array3D.h>

#include <vector>
#include <Python.h>
#include <CompuCell3D/Field3D/Field3D.h>

#include "FiPyInterfaceDLLSpecifier.h"

// template <typename Y> class Field3D;
typedef float precision_t;

namespace CompuCell3D{
  class FIPYINTERFACEEXTRACTOR_EXPORT  FiPyInterfaceBase{
    public:
      FiPyInterfaceBase(int _dim);
      ~FiPyInterfaceBase();
      
      void (FiPyInterfaceBase::*dimPtr)(void);
      void test1();
      void test2();
      void fillArray3D(PyObject * _FiPyArray,CompuCell3D::Field3D<float>* _Field);
      void ExampleFunc(PyObject * _FiPyArray,CompuCell3D::Field3D<float>* _Field);
      void doNotDiffuseList(vector<int>& _vec);
      std::vector<std::vector<int> > getDoNoDiffuseVec();
      void test(std::vector<int>& v);
      
      
      std::vector<std::vector<int> > doNotDiffuseVec;
  };
};


/*
#include <CompuCell3D/Field3D/Field3D.h>

template <typename Y> class Field3D;
typedef float precision_t;*/

// #include <iostream>
// #include <CompuCell3D/Field3D/Field3D.h>
// #include <CompuCell3D/Field3D/Array3D.h>

// void TestFuncOutSideClass();

// #include <CompuCell3D/Field3D/Field3D.h>
// #include <Python.h>
// #include <numpy/arrayobject.h>

/*class Zoo{
    public:
      Zoo();
      void test1();
      void test2();
      void fillArray3D(PyObject * _FiPyArray);

};*/


// #include <vector>
// #include <iostream>

// #include <python2.7/Python.h>
// #include <numpy/arrayobject.h>



// #include <CompuCell3D/Field3D/Array3D.h>

// void test() {}

//   template <typename Y> class Field3D;
//   template <typename Y> class Field3DImpl;
//   typedef float precision_t;
// 
//   void fillArray3D(Array3DFiPy<precision_t> *Field, PyObject * _FiPyArray) {
//     cout << "Hello World\n";
//   }

  
// }




#endif