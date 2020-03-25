

#ifndef FORCECALCULATOR_H

#define FORCECALCULATOR_H



#include <CompuCell3D/CC3D.h>


#include "CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h"
#include "CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h"




#include "ForceCalculatorDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

    

  template <class T> class Field3DImpl;
  template <class T> class WatchableField3D;



    class Potts3D;
    class Automaton;
    class BoundaryStrategy;
    class CellInventory;
    class CellG;

	class BoundaryPixelTrackerData;
	class BoundaryPixelTrackerPlugin;
  
	class FORCECALCULATOR_EXPORT ForceFieldData {
	public:
		std::vector<float> Force;

		ForceFieldData() {
			Force = std::vector<float>(3, 0.0);
		};
		// virtual ~ForceFieldData() {}

		std::vector<float> getForce() { return Force; }
		void setForce(std::vector<float> force) { Force = force; }
		
	};

  class FORCECALCULATOR_EXPORT ForceCalculator : public Steppable {

    WatchableField3D<CellG *> *cellFieldG;

    Simulator * sim;

    Potts3D *potts;

    CC3DXMLElement *xmlData;

    Automaton *automaton;

    BoundaryStrategy *boundaryStrategy;

    CellInventory * cellInventoryPtr;

	BoundaryPixelTrackerPlugin *boundaryTrackerPlugin;
    
	int neighborOrder;

    Dim3D fieldDim;
    
    int maxDimIdx;

	Field3DImpl<ForceFieldData* > *ForceField;

	void ForceCalculator::InitializeForceField(Dim3D fieldDim);
	BasicClassAccessor<ForceFieldData> ForceFieldDataAccessor;

	void deleteForceField();

  public:

    ForceCalculator ();

    virtual ~ForceCalculator ();

    // SimObject interface

	BasicClassAccessor<ForceFieldData> * getForceFieldDataAccessorPtr() { return &ForceFieldDataAccessor; }

    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    virtual void extraInit(Simulator *simulator);

	float getForceComponent(Point3D &pt, unsigned int compIdx);
    

    

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

