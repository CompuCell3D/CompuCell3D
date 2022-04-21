#ifndef NEIGHBORTRACKERPLUGIN_H
#define NEIGHBORTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "NeighborTracker.h"
#include "NeighborTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

    class Cell;

    class Field3DIndex;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class CellInventory;

    class BoundaryStrategy;

    class NEIGHBORTRACKER_EXPORT NeighborTrackerPlugin : public Plugin, public CellGChangeWatcher {

        ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
        WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <NeighborTracker> neighborTrackerAccessor;
        Simulator *simulator;
        bool periodicX, periodicY, periodicZ;
        CellInventory *cellInventoryPtr;
        bool checkSanity;
        unsigned int checkFreq;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;


    public:
        NeighborTrackerPlugin();

        virtual ~NeighborTrackerPlugin();


        // Field3DChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        //Plugin interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual std::string toString();


        ExtraMembersGroupAccessor <NeighborTracker> *
        getNeighborTrackerAccessorPtr() { return &neighborTrackerAccessor; }

        // End XMLSerializable interface
        int returnNumber() { return 23432; }

        short getCommonSurfaceArea(NeighborSurfaceData *_nsd) { return _nsd->commonSurfaceArea; }


    protected:
        double distance(double, double, double, double, double, double);

        virtual void testLatticeSanityFull();

        bool isBoundaryPixel(Point3D pt);

        bool watchingAllowed;
        AdjacentNeighbor adjNeighbor;
        long maxIndex; //maximum field index
        long changeCounter;
    };
};
#endif
