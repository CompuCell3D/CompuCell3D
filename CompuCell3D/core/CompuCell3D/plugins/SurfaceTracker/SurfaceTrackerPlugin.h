

#ifndef SURFACETRACKERPLUGIN_H
#define SURFACETRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

// // // #include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>
#include "SurfaceTrackerDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {
  
  
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class BoundaryStrategy;
  class Potts3D;
	
  class SURFACETRACKER_EXPORT SurfaceTrackerPlugin : public Plugin, public CellGChangeWatcher {


    WatchableField3D<CellG *> *cellFieldG;
    unsigned int maxNeighborIndex;
    BoundaryStrategy *boundaryStrategy;
    LatticeMultiplicativeFactors lmf;
    Potts3D *potts;

  public:

    SurfaceTrackerPlugin();
    virtual ~SurfaceTrackerPlugin();

  

    // SimObject interface
	 virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    const LatticeMultiplicativeFactors & getLatticeMultiplicativeFactors() const {return lmf;}
    unsigned int getMaxNeighborIndex(){return maxNeighborIndex;}

    virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);
			       
			       
    // Begin XMLSerializable interface
    //virtual void readXML(XMLPullParser &in);
    //virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
    //SteerableObject interface
	 virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

	 

    virtual std::string steerableName();
	 virtual std::string toString();

  };
};
#endif
