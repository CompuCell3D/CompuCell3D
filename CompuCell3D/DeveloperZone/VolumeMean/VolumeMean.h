

#ifndef VOLUMEMEANSTEPPABLE_H
#define VOLUMEMEANSTEPPABLE_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Steppable.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <vector>

#include "VolumeMeanDLLSpecifier.h"

namespace CompuCell3D {


  template <class T> class Field3D;
  template <class T> class WatchableField3D;

  class Potts3D;

  class VOLUMEMEAN_EXPORT VolumeMean : public Steppable {

    WatchableField3D<CellG *> *cellFieldG;
    Simulator * sim;
    Potts3D *potts;
    double exponent;



  public:
    VolumeMean();

    virtual ~VolumeMean();
    // SimObject interface
	 virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
    virtual void extraInit(Simulator *simulator);


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
