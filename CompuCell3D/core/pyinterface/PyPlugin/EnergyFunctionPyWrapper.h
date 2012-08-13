#ifndef ENERGYFUNCTIONPYWRAPPER_H
#define ENERGYFUNCTIONPYWRAPPER_H

#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include "PyCompuCellObjAdapter.h"

namespace CompuCell3D{
   class Simulator;
   class Potts3D;

   class EnergyFunctionPyWrapper:public PyCompuCellObjAdapter, public EnergyFunction{

   private:
	   //might need to introduce one lock for all  PyWrapper classes (stored in ParallelUtilsOpenMp) so that e.g. changeWatcher cannot run concurently with energy function
	   // but for now GIL should do the trick. It might cause some problems with performance. 
	   //However, because change watchers/energy cal;culators are rarely done in Python this sould not be that big of a problem


	   ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
   
   public:
	  EnergyFunctionPyWrapper();
	  virtual ~EnergyFunctionPyWrapper();
	  
	  EnergyFunction * getEnergyFunctionPyWrapperPtr();
     virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
            const CellG *oldCell);
     virtual double localEnergy(const Point3D &pt);
     void registerPyEnergyFunction(PyObject * _enFunction);

   };



};



#endif
