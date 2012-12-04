#ifndef CENTERMODELOBJECT_H
#define CENTERMODELOBJECT_H

#include "InterfacesDLLSpecifier.h"
#include <string>
#include <iostream>

#include <Components/CellCM.h>
#include <Components/Interfaces/Steerable.h>

namespace CenterModel {

	class SimulationBox;
    class SimulatorCM;

	class INTERFACES_EXPORT CenterModelObject: public Steerable{
    
	public:
		       
		CenterModelObject():sbPtr(0),simulator(0),xmlData(0){};

		virtual ~CenterModelObject(){};
        
        //ForceTerm interface
        virtual void init(SimulatorCM *_simulator=0, CC3DXMLElement * _xmlData=0)=0;
        
		virtual std::string getName()=0;
        

	protected:
        SimulationBox *sbPtr;
        SimulatorCM * simulator;
        CC3DXMLElement *xmlData;

	};

};
#endif
