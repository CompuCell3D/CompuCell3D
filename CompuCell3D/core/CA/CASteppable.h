#ifndef PROBABILITYFUNCTION_H
#define PROBABILITYFUNCTION_H

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include "SimulationObject.h"

namespace CompuCell3D{

  class Point3D;  
  class CAManager;
  class CACell;
  class CACellStack;
  
  template<typename T>
  class Field3D;
  
  class CASteppable :public SimulationObject {
  public:
    int frequency;

    CASteppable() : frequency(1) {}
    virtual ~CASteppable() {}

    virtual void start() {};
    virtual void step(const unsigned int currentStep) {};
    virtual void finish() {};
    virtual std::string toString(){return "Steppable";}

  };
};

#endif
