

#ifndef SIMPLEVOLUMEPLUGIN_H
#define SIMPLEVOLUMEPLUGIN_H

#include <CompuCell3D/Plugin.h>
#include <CompuCell3D/Potts3D/Stepper.h>
#include <CompuCell3D/Potts3D/EnergyFunction.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include "SimpleVolumeDLLSpecifier.h"
#include <vector>
#include <string>

class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class CellG;

  class SIMPLEVOLUME_EXPORT SimpleVolumePlugin : public Plugin , public EnergyFunction 
  {
	Potts3D *potts;
	CC3DXMLElement *xmlData;


  public:
    double targetVolume;
    double lambdaVolume;

    SimpleVolumePlugin():potts(0){};
    virtual ~SimpleVolumePlugin(){};

	//EnergyFunction interface
	virtual double changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

    // SimObject interface
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	virtual std::string toString();
  };
};
#endif
