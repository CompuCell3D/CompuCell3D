

#ifndef PRESSURECALCULATORSTEPPABLE_H

#define PRESSURECALCULATORSTEPPABLE_H



#include <CompuCell3D/CC3D.h>



#include "PressureCalculatorData.h"



#include "PressureCalculatorDLLSpecifier.h"





namespace CompuCell3D {

    

  template <class T> class Field3D;

  template <class T> class WatchableField3D;



    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class CellInventory;

    class CellG;

  

  class PRESSURECALCULATOR_EXPORT PressureCalculator : public Steppable {



    BasicClassAccessor<PressureCalculatorData> pressureCalculatorDataAccessor;                

    WatchableField3D<CellG *> *cellFieldG;

    Simulator * sim;

    Potts3D *potts;

    CC3DXMLElement *xmlData;

    Automaton *automaton;

    BoundaryStrategy *boundaryStrategy;

    CellInventory * cellInventoryPtr;

    

    Dim3D fieldDim;

	typedef double (PressureCalculator::*pressureCalc_t)(const CellG *cell);
	PressureCalculator::pressureCalc_t pressureCalcFcnPtr;

	double pressureCalcGlobal(const CellG *cell);
	double pressureCalcByType(const CellG *cell);
	double pressureCalcByID(const CellG *cell);


    

  public:

    PressureCalculator ();

    virtual ~PressureCalculator ();

    // SimObject interface

    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    virtual void extraInit(Simulator *simulator);



    BasicClassAccessor<PressureCalculatorData> * getPressureCalculatorDataAccessorPtr(){return & pressureCalculatorDataAccessor;}

    

    //steppable interface

    virtual void start();

    virtual void step(const unsigned int currentStep);

    virtual void finish() {}





    //SteerableObject interface

    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

	virtual double pressureCalc(const CellG *cell);

    virtual std::string steerableName();

	 virtual std::string toString();



  };

};

#endif        

