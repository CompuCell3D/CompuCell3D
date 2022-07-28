
#ifndef REALPLASTICITYPLUGIN_H
#define REALPLASTICITYPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "PlasticityDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {


    template<class T>
    class Field3D;

    class Point3D;

    class Simulator;

    class PlasticityTrackerData;

    class PlasticityTracker;

    class BoundaryStrategy;

    /**
     * Calculates surface energy based on a target surface and
     * lambda surface.
     */
    class BoundaryStrategy;

    class PLASTICITY_EXPORT PlasticityPlugin : public Plugin, public EnergyFunction {


        Field3D<CellG *> *cellFieldG;
        std::string pluginName;

        //energy function data

        float targetLengthPlasticity;
        float maxLengthPlasticity;
        double lambdaPlasticity;
        Simulator *simulator;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <PlasticityTracker> *plasticityTrackerAccessorPtr;

        typedef double (PlasticityPlugin::*diffEnergyFcnPtr_t)(float _deltaL, float _lBefore,
                                                               const PlasticityTrackerData *_plasticityTrackerData,
                                                               const CellG *_cell);

        diffEnergyFcnPtr_t diffEnergyFcnPtr;
        BoundaryStrategy *boundaryStrategy;


    public:
        PlasticityPlugin();

        virtual ~PlasticityPlugin();

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual std::string toString();

        virtual void extraInit(Simulator *simulator);

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        //EnergyFunction methods
        double diffEnergyGlobal(float _deltaL, float _lBefore, const PlasticityTrackerData *_plasticityTrackerData,
                                const CellG *_cell);

        double diffEnergyLocal(float _deltaL, float _lBefore, const PlasticityTrackerData *_plasticityTrackerData,
                               const CellG *_cell);


    };
};
#endif
