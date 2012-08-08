#ifndef CHANGEWATCHERPYWRAPPER_H
#define CHANGEWATCHERPYWRAPPER_H

#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include "PyCompuCellObjAdapter.h"

namespace CompuCell3D{
   class Simulator;
   class Potts3D;

   class ChangeWatcherPyWrapper:public PyCompuCellObjAdapter, public CellGChangeWatcher{
   private:
	   //might need to introduce one lock for all  PyWrapper classes (stored in ParallelUtilsOpenMp) so that e.g. changeWatcher cannot run concurently with energy function
	   // but for now GIL should do the trick. It might cause some problems with performance. 
	   //However, because change watchers/energy cal;culators are rarely done in Python this sould not be that big of a problem

	   ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;   
   
   public:
	  ChangeWatcherPyWrapper();
	  ~ChangeWatcherPyWrapper();
	  virtual void field3DChange(const Point3D &pt, CellG *_newCell, CellG *_oldCell);
	  CellGChangeWatcher * getChangeWatcherPyWrapperPtr();
     void registerPyChangeWatcher(PyObject * _fieldWatcher);
   };



};



#endif
