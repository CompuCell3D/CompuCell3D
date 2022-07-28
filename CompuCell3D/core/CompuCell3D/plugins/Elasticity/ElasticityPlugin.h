

#ifndef ELASTICITYPLUGIN_H
#define ELASTICITYPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "ElasticityDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {


    template<class T>
    class Field3D;

    class Point3D;

    class Simulator;

    class ElasticityTrackerData;

    class ElasticityTracker;

    class BoundaryStrategy;

    /**
     * Calculates surface energy based on a target surface and
     * lambda surface.
     */
    class BoundaryStrategy;

    class ELASTICITY_EXPORT ElasticityPlugin : public Plugin, public EnergyFunction {


        Field3D<CellG *> *cellFieldG;
        std::string pluginName;
        //energy function data

        float targetLengthElasticity;
        float maxLengthElasticity;
        double lambdaElasticity;
        Simulator *simulator;
        Potts3D *potts;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <ElasticityTracker> *elasticityTrackerAccessorPtr;

        typedef double (ElasticityPlugin::*diffEnergyFcnPtr_t)(float _deltaL, float _lBefore,
                                                               const ElasticityTrackerData *_elasticityTrackerData,
                                                               const CellG *_cell);

        diffEnergyFcnPtr_t diffEnergyFcnPtr;
        BoundaryStrategy *boundaryStrategy;


    public:
        ElasticityPlugin();

        virtual ~ElasticityPlugin();

        //Plugin interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual std::string toString();

        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        //EnergyFunction methods
        double diffEnergyGlobal(float _deltaL, float _lBefore, const ElasticityTrackerData *_elasticityTrackerData,
                                const CellG *_cell);

        double diffEnergyLocal(float _deltaL, float _lBefore, const ElasticityTrackerData *_elasticityTrackerData,
                               const CellG *_cell);


    };
};
#endif
