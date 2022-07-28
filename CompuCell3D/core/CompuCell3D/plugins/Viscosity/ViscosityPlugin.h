#ifndef VISCOSITYPLUGIN_H
#define VISCOSITYPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ViscosityDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class CellG;

    class Simulator;

    class NeighborTracker;

    class VISCOSITY_EXPORT ViscosityPlugin : public Plugin, public EnergyFunction/*,public CellGChangeWatcher*/ {

    private:
        Potts3D *potts;
        CC3DXMLElement *xmlData;
        Simulator *sim;
        ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr;
        Point3D boundaryConditionIndicator;
        Dim3D fieldDim;
        BoundaryStrategy *boundaryStrategy;
        unsigned int maxNeighborIndex;
        double lambdaViscosity;
        std::string pluginName;

        double dist(double _x, double _y, double _z);

    public:
        ViscosityPlugin();

        virtual ~ViscosityPlugin();

        // SimObject interface
        virtual void extraInit(Simulator *simulator);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif
