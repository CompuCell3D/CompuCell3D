#ifndef GROWTHSTEPPABLESTEPPABLE_H
#define GROWTHSTEPPABLESTEPPABLE_H
#include <CompuCell3D/CC3D.h>
#include "GrowthSteppableDLLSpecifier.h"

namespace CompuCell3D {
   
    template <class T> class Field3D;
    template <class T> class WatchableField3D;

    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class CellInventory;
    class CellG;

  

  class GROWTHSTEPPABLE_EXPORT GrowthSteppable : public Steppable {

    WatchableField3D<CellG *> *cellFieldG;

    Simulator * sim;

    Potts3D *potts;

    CC3DXMLElement *xmlData;

    Automaton *automaton;

    BoundaryStrategy *boundaryStrategy;

    CellInventory * cellInventoryPtr;
    
    Dim3D fieldDim;
   
  public:
    
    GrowthSteppable ();

    virtual ~GrowthSteppable ();

    std::map<unsigned int, double> growthRateMap;

    // SimObject interface

    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    virtual void extraInit(Simulator *simulator);

    // Python wrapper functions

    void setGrowthRate(unsigned int cellType, double growthRate);

    //steppable interface

    virtual void start();

    virtual void step(const unsigned int currentStep);

    virtual void finish() {}


    //SteerableObject interface

    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

    virtual std::string steerableName();

    virtual std::string toString();

  };

};

#endif        

