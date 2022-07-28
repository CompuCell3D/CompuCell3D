#ifndef CLUSTERSURFACETRACKERPLUGIN_H
#define CLUSTERSURFACETRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "ClusterSurfaceTrackerDLLSpecifier.h"


class CC3DXMLElement;

namespace CompuCell3D {

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class Simulator;

    class Potts3D;

    class Automaton;

    //class AdhesionFlexData;
    class BoundaryStrategy;

    class ParallelUtilsOpenMP;

    class PixelTracker;

    class PixelTrackerPlugin;

    class PixelTrackerData;


    class CLUSTERSURFACETRACKER_EXPORT  ClusterSurfaceTrackerPlugin : public Plugin, public CellGChangeWatcher {

    private:

        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        LatticeMultiplicativeFactors lmf;
        WatchableField3D<CellG *> *cellFieldG;
        unsigned int maxNeighborIndex;

        PixelTrackerPlugin *pixelTrackerPlugin;
        ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr;

    public:

        ClusterSurfaceTrackerPlugin();

        virtual ~ClusterSurfaceTrackerPlugin();


        // CellChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

        const LatticeMultiplicativeFactors &getLatticeMultiplicativeFactors() const { return lmf; }

        unsigned int getMaxNeighborIndex() { return maxNeighborIndex; }

        void updateClusterSurface(long _clusterId);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        //Steerrable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif
        
