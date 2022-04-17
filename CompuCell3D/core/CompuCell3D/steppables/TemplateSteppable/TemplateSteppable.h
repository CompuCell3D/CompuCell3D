

#ifndef TEMPLATESTEPPABLE_H
#define TEMPLATESTEPPABLE_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Steppable.h>

// // // #include <string>

// // // template <typename Y> class BasicClassAccessor;
#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>
#include "TemplateSteppableDLLSpecifier.h"

namespace CompuCell3D {
  class Potts3D;
  class CellInventory;
  class BoundaryStrategy;

  class TEMPLATESTEPPABLE_EXPORT TemplateSteppable : public Steppable {
    Potts3D *potts;

    BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
    std::string pifname;

  public:


    CellInventory * cellInventoryPtr;

    BoundaryStrategy * boundaryStrategy;
    
    TemplateSteppable();
    TemplateSteppable(std::string);

    void setPotts(Potts3D *potts) {this->potts = potts;}

    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    // Begin Steppable interface
    virtual void start();
    virtual void step(const unsigned int currentStep); 
    virtual void finish() {}
    // End Steppable interface


  };
};
#endif
