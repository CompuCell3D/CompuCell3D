#ifndef SIMULATIONOBJECT_H
#define SIMULATIONOBJECT_H

#include <string>
namespace CompuCell3D{


  class CAManager;
  
  
  class SimulationObject {
  


    public:
        SimulationObject(){}
		virtual ~SimulationObject(){}
        virtual void init(CAManager *_caManager){}
		virtual void extraInit(){}
        virtual void extraInit2(){}
		virtual std::string toString(){return "SimulationObject";}
  };
};

#endif
