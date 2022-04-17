

#ifndef STRECHNESSPLUGIN_H
#define STRECHNESSPLUGIN_H

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Plugin.h>

// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>
// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>

#include "StretchnessDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
  

  template <class T> class Field3D;
  template <class T> class WatchableField3D;

  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  class Point3D;

  class BoundaryStrategy;

  class STRETCHNESS_EXPORT StretchnessPlugin : public Plugin, public EnergyFunction{
		CC3DXMLElement *xmlData;
	  //EnergyFunction Data	
	 WatchableField3D<CellG *> *cellFieldG;


    double targetStretchness;
    double lambdaStretchness;
    double scaleSurface;
    BoundaryStrategy *boundaryStrategy;
    unsigned int maxNeighborIndex;
    LatticeMultiplicativeFactors lmf;


  public:
    StretchnessPlugin();
    virtual ~StretchnessPlugin();

  
		//EnergyFunction Interface
		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

	   // SimObject interface
		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
		virtual void extraInit(Simulator *simulator);



    //Steerrable interface
	 virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
	 virtual std::string steerableName();

	 virtual std::string toString();

	 //EnergyFunction methods
    double diffEnergy(double surface, double diff);
    
    



  };
};
#endif
