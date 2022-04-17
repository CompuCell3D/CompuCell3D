

#ifndef REARRANGEMENTPLUGIN_H
#define REARRANGEMENTPLUGIN_H
#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>


// // // #include <CompuCell3D/Potts3D/Stepper.h>
// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>


// // // #include <CompuCell3D/Potts3D/Cell.h>

#include "RearrangementDLLSpecifier.h"

namespace CompuCell3D {


  	class BoundaryStrategy;
   class Potts3D;
   class Simulator;
   class CellG;
   template <class T> class Field3D;
   template <class T> class WatchableField3D;
   
	class NeighborTracker;
  

  
	class REARRANGEMENT_EXPORT  RearrangementPlugin : public Plugin,public EnergyFunction {
        
    

    Potts3D *potts;
	//energyFunction data

    double fRearrangement;
    double lambdaRearrangement;
    unsigned int maxNeighborIndex;
    BoundaryStrategy *boundaryStrategy;
    WatchableField3D<CellG *> *cellFieldG;
    BasicClassAccessor<NeighborTracker> * neighborTrackerAccessorPtr;
    float percentageLossThreshold;
    float defaultPenalty;


  public:
    RearrangementPlugin();
    virtual ~RearrangementPlugin();
    
    // SimObject interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
	//energy Function interface
    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);


    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	 virtual std::string toString();

	 //energy function methods
   double getLambdaRearrangement()const{return lambdaRearrangement;}
   double getFRearrangement()const{return fRearrangement;}
	std::pair<CellG*,CellG*> preparePair(CellG* , CellG*);


  };
};
#endif
