#ifndef COMPUCELL3DCELLVELOCITY_H
#define COMPUCELL3DCELLVELOCITY_H

#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <Utils/cldeque.h>

template <typename Y> class BasicClassAccessor;

namespace CompuCell3D {

/**
@author m
*/
class Potts3D;
class CellInventory;
class CellVelocityData;


class CellVelocity : public Steppable
{
   
   
   public:
      CellVelocity();

      virtual ~CellVelocity();
    
      // SimObject interface
      virtual void init(Simulator *simulator);
      virtual void extraInit(Simulator *simulator);
      // Begin Steppable interface
      virtual void start();
      virtual void step(const unsigned int currentStep);
      virtual void finish() {}
      // End Steppable interface

      // Begin XMLSerializable interface
      virtual void readXML(XMLPullParser &in);
      virtual void writeXML(XMLSerializer &out){};
      // End XMLSerializable interface
    protected:
      Potts3D *potts;
      CellInventory *cellInventoryPtr;
      BasicClassAccessor<CellVelocityData> * cellVelocityDataAccessorPtr;
/*      unsigned int numberOfCOMPoints;
      unsigned int enoughDataThreshold;*/
      unsigned int updateFrequency;
      void resizeCellVelocityData();
      void updateCOMList();
      void zeroCellVelocities();

      Point3D boundaryConditionIndicator;
      Dim3D fieldDim;
};

};





#endif
