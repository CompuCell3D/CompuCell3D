#ifndef STEPPERPYWRAPPER_H
#define STEPPERPYWRAPPER_H

#include <CompuCell3D/Potts3D/Stepper.h>
#include "PyCompuCellObjAdapter.h"

namespace CompuCell3D{
   class Simulator;
   class Potts3D;

   class StepperPyWrapper:public PyCompuCellObjAdapter, public Stepper{
   private:
	   //might need to introduce one lock for all  PyWrapper classes (stored in ParallelUtilsOpenMp) so that e.g. changeWatcher cannot run concurently with energy function
	   // but for now GIL should do the trick. It might cause some problems with performance. 
	   //However, because change watchers/energy cal;culators are rarely done in Python this sould not be that big of a problem

	   ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;     
   public:
	  StepperPyWrapper();
	  ~StepperPyWrapper();
	  virtual void step();
	  Stepper* getStepperPyWrapperPtr();
     void registerPyStepper(PyObject * _pyStepper);
   };



};



#endif
