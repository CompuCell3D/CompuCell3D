#ifndef COMPUCELL3DCELLVELOCITYPLUGIN_H
#define COMPUCELL3DCELLVELOCITYPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <BasicUtils/BasicClassAccessor.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>

namespace CompuCell3D {

/**
@author m
*/

    class Simlator;

    class Potts3D;

    class CenterOfMassPlugin;

    class CellVelocityPlugin : public Plugin
// ,public CellGChangeWatcher

    {
    public:
        CellVelocityPlugin();

        virtual ~CellVelocityPlugin();

        BasicClassAccessor <CellVelocityData> *getCellVelocityDataAccessorPtr() { return cellVelocityDataAccessorPtr; }

        // SimObject interface
        virtual void init(Simulator *_simulator);

        virtual void extraInit(Simulator *_simulator);

        // CellGChangeWatcher interface
//       virtual void field3DChange(const Point3D &pt, CellG *newCell,
//                                CellG *oldCell);

        // Begin XMLSerializable interface
        virtual std::string toString() { return "CellVelocity"; }

        // End XMLSerializable interface
        cldeque<Coordinates3D < float> >

        ::size_type getCldequeCapacity() { return cldequeCapacity; }

    protected:

        BasicClassAccessor <CellVelocityData> *cellVelocityDataAccessorPtr;
        cldeque<Coordinates3D < float> >
        ::size_type cldequeCapacity;
        cldeque<Coordinates3D < float> >
        ::size_type enoughDataThreshold;


    };


};

#endif
