

#ifndef SIMPLEARRAYPLUGIN_H
#define SIMPLEARRAYPLUGIN_H

#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

#include <CompuCell3D/Plugin.h>
#include "SimpleArray.h"


#include <vector>

#include <CompuCell3D/dllDeclarationSpecifier.h>

namespace CompuCell3D {
  class Potts3D;

  class Cell;
  class CellInventory;
  
  template <typename Y> class Field3DImpl;
  
  class DECLSPECIFIER SimpleArrayPlugin : public Plugin {
     
     BasicClassAccessor<SimpleArray> simpleArrayAccessor;
     CellInventory * cellInventoryPtr;
     Field3D<float> *simpleArrayFieldPtr;
     Potts3D *potts;
   

  public:
    SimpleArrayPlugin();
    virtual ~SimpleArrayPlugin();

    BasicClassAccessor<SimpleArray> * getSimpleArrayAccessorPtr(){return &simpleArrayAccessor;}
    // SimObject interface
    virtual void init(Simulator *_simulator, ParseData *_pd=0);
    virtual void extraInit(Simulator *_simulator);
    void update(ParseData *_pd);

    
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    virtual std::string toString(){return "SimpleArray";}
    
    // End XMLSerializable interface
     protected:
        std::vector<double> probMatrix;
   
  };
};
#endif
