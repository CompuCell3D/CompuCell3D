#ifndef VELOCITYPLUGIN_H
#define VELOCITYPLUGIN_H

#include <CompuCell3D/Plugin.h>
//#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/plugins/Velocity/VelocityData.h>
#include <CompuCell3D/Potts3D/EnergyFunction.h>

namespace CompuCell3D {

/**
@author m
*/
class Potts3D;
class Simulator;

class VelocityPlugin : public Plugin, public CellGChangeWatcher,public EnergyFunction

{
   BasicClassAccessor<VelocityData> velocityDataAccessor;

   public:
      VelocityPlugin();
      
      virtual ~VelocityPlugin();
      virtual void init(Simulator *_simulator);
      virtual void extraInit(Simulator *_simulator);

      
      // CellGChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                               CellG *oldCell);
     //no energy is calculated rather we use this API to precalculate cellCM and cellVelocity after potential spin flip
     virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
				const CellG *oldCell);
      virtual double localEnergy(const Point3D & pt);
      virtual std::string toString(){return "Velocity";}
    
      BasicClassAccessor<VelocityData> * getVelocityDataAccessorPtr(){return &velocityDataAccessor;}
      void calculateVelocityData(const Point3D &pt, const CellG *newCell,const CellG *oldCell);

      // Begin XMLSerializable interface
      virtual void readXML(XMLPullParser &in);
      virtual void writeXML(XMLSerializer &out);
    // End XMLSerializable interface
      
      
      
   private:
      Simulator *sim;
      Potts3D *potts;
      Dim3D fieldDim;
      Point3D boundaryConditionIndicator;
      
      
};


};

#endif
