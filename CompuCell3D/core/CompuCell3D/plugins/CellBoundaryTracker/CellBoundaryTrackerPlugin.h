

#ifndef CELLBOUNDARYTRACKERPLUGIN_H
#define CELLBOUNDARYTRACKERPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include "CellBoundaryTracker.h"
#include <CompuCell3D/Field3D/AdjacentNeighbor.h>

namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;


  class CellBoundaryTrackerPlugin : public Plugin, public CellGChangeWatcher {
    //CellBoundaryDynamicClassNode classNode; //will have to register it with cellFactory from Potts3D
//    SurfaceEnergy *surfaceEnergy;

    Field3D<CellG *> *cellFieldG;
    Dim3D fieldDim;
    //CellBoundaryTracker cellBoundaryTracker;
    BasicClassAccessor<CellBoundaryTracker> cellBoundaryTrackerAccessor;
    Simulator *simulator;
    bool periodicX,periodicY,periodicZ;
    
  public:
    CellBoundaryTrackerPlugin();
    virtual ~CellBoundaryTrackerPlugin();

    //CellBoundaryDynamicClassNode getClassNode() {return classNode;}

    // SimObject interface
    virtual void init(Simulator *simulator);

    // BCGChangeWatcher interface
    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);

    // Begin XMLSerializable interface
    virtual void readXML(XMLPullParser &in);
    virtual void writeXML(XMLSerializer &out);
    virtual void initializeBoundaries()  ;
    BasicClassAccessor<CellBoundaryTracker> * getCellBoundaryTrackerAccessorPtr(){return & cellBoundaryTrackerAccessor;}
    // End XMLSerializable interface
    
   protected:
   double distance(double,double,double,double,double,double);
   
   virtual void testLatticeSanity();
   virtual void testLatticeSanityFull();
   bool isBoundaryPixel(Point3D pt);
   bool isTouchingLatticeBoundary(Point3D pt,Point3D ptAdj);
   bool watchingAllowed;
   AdjacentNeighbor adjNeighbor;
   long maxIndex; //maximum field index     
   long changeCounter;
  };
};
#endif
