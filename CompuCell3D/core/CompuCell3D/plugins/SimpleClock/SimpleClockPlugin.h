

#ifndef SIMPLECLOCKPLUGIN_H
#define SIMPLECLOCKPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>


// // // #include <CompuCell3D/Potts3D/Stepper.h>

// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>
#include "SimpleClock.h"


#include "SimpleClockDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;

  class Cell;
  template <typename Y> class Field3DImpl;
  
  class SIMPLECLOCK_EXPORT SimpleClockPlugin : public Plugin
  //, public CellGChangeWatcher,public Stepper 
		       {
    
    ExtraMembersGroupAccessor<SimpleClock> simpleClockAccessor;
        
    
    Field3D<float> *simpleClockFieldPtr;
    Potts3D *potts;
        
	// Point3D pt;
  public:

    SimpleClockPlugin();
    virtual ~SimpleClockPlugin();
	 
    ExtraMembersGroupAccessor<SimpleClock> * getSimpleClockAccessorPtr(){return &simpleClockAccessor;}
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    void setSimpleClockFieldPtr( Field3D<float> *_simpleClockFieldPtr){simpleClockFieldPtr=_simpleClockFieldPtr;}   

  };
};
#endif
