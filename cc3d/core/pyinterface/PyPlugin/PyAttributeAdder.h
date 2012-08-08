#ifndef PYATTRIBUTEADDER_H
#define PYATTRIBUTEADDER_H

#include <CompuCell3D/Potts3D/AttributeAdder.h>
#include "PyCompuCellObjAdapter.h"

namespace CompuCell3D{

class PyAttributeAdder :public PyCompuCellObjAdapter , public AttributeAdder{
    public:
      PyAttributeAdder():refChecker(0),destroyer(0){}
      virtual void addAttribute(CellG *);
      virtual void destroyAttribute(CellG *);
      AttributeAdder * getPyAttributeAdderPtr();
      void registerAdder(PyObject *);
      void registerRefChecker(PyObject * _refChecker){refChecker=_refChecker;}
      void registerDestroyer(PyObject * _destroyer){destroyer=_destroyer;}

      PyObject * refChecker;
      PyObject * destroyer;
};


};
#endif
