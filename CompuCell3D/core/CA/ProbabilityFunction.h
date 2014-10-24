#ifndef PROBABILITYFUNCTION_H
#define PROBABILITYFUNCTION_H

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>


namespace CompuCell3D{

  class Point3D;  
  class CAManager;
  class CACell;
  class CACellStack;
  
  template<typename T>
  class Field3D;
  
  class ProbabilityFunction {
  
    protected:
    
        CAManager *caManager;
        Field3D<CACellStack *> *cellFieldS;    
        Dim3D fieldDim;
        
    public:
        ProbabilityFunction():caManager(0){}
        virtual void init(CAManager *_caManager){}
        virtual std::string toString(){return "ProbabilityFunction";}
        virtual float calculate(const Point3D & _source, const Point3D & _target){return 1.0;};
        virtual ~ProbabilityFunction(){}
  };
};

#endif
