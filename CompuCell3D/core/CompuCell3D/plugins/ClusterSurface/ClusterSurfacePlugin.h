#ifndef CLUSTERSURFACEPLUGIN_H
#define CLUSTERSURFACEPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "ClusterSurfaceDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;

    class Automaton;

    //class AdhesionFlexData;
    class BoundaryStrategy;

    class ClusterSurfaceTrackerPlugin;

    class ParallelUtilsOpenMP;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class CLUSTERSURFACE_EXPORT  ClusterSurfacePlugin : public Plugin, public EnergyFunction {

    private:

        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;

        LatticeMultiplicativeFactors lmf;
        unsigned int maxNeighborIndex;

        enum FunctionType {
            GLOBAL = 0, BYCELLTYPE = 1, BYCELLID = 2
        };
        FunctionType functionType;

        double targetClusterSurface;
        double lambdaClusterSurface;

        double scaleClusterSurface;

        typedef double (ClusterSurfacePlugin::*changeEnergy_t)(const Point3D &pt, const CellG *newCell,
                                                               const CellG *oldCell);

        ClusterSurfaceTrackerPlugin *cstPlugin;

        ClusterSurfacePlugin::changeEnergy_t changeEnergyFcnPtr;

    public:

        ClusterSurfacePlugin();

        virtual ~ClusterSurfacePlugin();

        //Energy function interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyByCellId(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyGlobal(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        void setTargetAndLambdaClusterSurface(CellG *_cell, float _targetClusterSurface, float _lambdaClusterSurface);

        std::pair<float, float>
        getTargetAndLambdaVolume(const CellG *_cell) const; //(targetClusterSurface,lambdaClusterSurface)

        std::pair<double, double>
        getNewOldClusterSurfaceDiffs(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double diffEnergy(double lambda, double targetSurface, double surface, double diff);


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        virtual void extraInit(Simulator *simulator);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
        
