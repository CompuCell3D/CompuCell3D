#ifndef COMPUCELL3DCELLINSTANTVELOCITYPLUGIN_H
#define COMPUCELL3DCELLINSTANTVELOCITYPLUGIN_H

#include <CompuCell3D/Plugin.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>

#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
#include <CompuCell3D/plugins/CellVelocity/InstantVelocityData.h>
#include <CompuCell3D/plugins/CellVelocity/CellVelocityPlugin.h>

namespace CompuCell3D {

/**
@author m
*/
    class Potts3D;

    class CellInstantVelocityPlugin : public CellVelocityPlugin, public CellGChangeWatcher {
    public:
        CellInstantVelocityPlugin();

        virtual ~CellInstantVelocityPlugin();

        virtual void init(Simulator *_simulator);

        virtual void extraInit(Simulator *_simulator);


        // CellGChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        virtual std::string toString() { return "CellInstantVelocity"; }


        InstantCellVelocityData &getInstantVelocityData() { return ivd; }

        void calculateInstantVelocityData(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


    private:

        Potts3D *potts;
        Dim3D fieldDim;
        Point3D boundaryConditionIndicator;
        InstantCellVelocityData ivd;

    };


};

#endif
