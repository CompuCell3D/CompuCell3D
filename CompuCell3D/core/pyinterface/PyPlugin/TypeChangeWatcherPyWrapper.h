#ifndef TYPECHANGEWATCHERPYWRAPPER_H
#define TYPECHANGEWATCHERPYWRAPPER_H

#include <CompuCell3D/Potts3D/TypeChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "PyCompuCellObjAdapter.h"

namespace CompuCell3D{
   class Simulator;
   class Potts3D;

   class TypeChangeWatcherPyWrapper:public PyCompuCellObjAdapter, public TypeChangeWatcher{

   private:
	   //might need to introduce one lock for all  PyWrapper classes (stored in ParallelUtilsOpenMp) so that e.g. changeWatcher cannot run concurently with energy function
	   // but for now GIL should do the trick. It might cause some problems with performance. 
	   //However, because change watchers/energy cal;culators are rarely done in Python this sould not be that big of a problem

	   ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
   
   public:
	  TypeChangeWatcherPyWrapper();
	  virtual ~TypeChangeWatcherPyWrapper();
	  virtual void typeChange(CellG* _cell,CellG::CellType_t _newType);
	  TypeChangeWatcher * getTypeChangeWatcherPyWrapperPtr();
     void registerPyTypeChangeWatcher(PyObject * _typeChangeWatcher);

//      unsigned char newType;
	 //CellG::CellType_t newType;

   };



};



#endif
