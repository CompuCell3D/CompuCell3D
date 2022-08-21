#ifndef GLOBALBOUNDARYPIXELTRACKERPLUGIN_H
#define GLOBALBOUNDARYPIXELTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "GlobalBoundaryPixelTrackerDLLSpecifier.h"
#include <unordered_set>

class CC3DXMLElement;
namespace CompuCell3D {

    class Cell;

    class Field3DIndex;

    class Potts3D;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class BoundaryStrategy;


    class GLOBALBOUNDARYPIXELTRACKER_EXPORT GlobalBoundaryPixelTrackerPlugin
            : public Plugin, public CellGChangeWatcher {

        ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        //WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        Simulator *simulator;
        Potts3D *potts;
        unsigned int maxNeighborIndex;
        float container_refresh_fraction;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;

        std::unordered_set <Point3D, Point3DHasher, Point3DComparator> *boundaryPixelSetPtr;
        std::unordered_set <Point3D, Point3DHasher, Point3DComparator> *justInsertedBoundaryPixelSetPtr;
        std::unordered_set <Point3D, Point3DHasher, Point3DComparator> *justDeletedBoundaryPixelSetPtr;
        std::vector <Point3D> *boundaryPixelVectorPtr;

        void insertPixel(Point3D &pt);

        void removePixel(Point3D &pt);

        void refreshContainers();


    public:
        GlobalBoundaryPixelTrackerPlugin();

        virtual ~GlobalBoundaryPixelTrackerPlugin();


        // Field3DChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        //Plugin interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *_simulators);

        virtual void handleEvent(CC3DEvent &_event);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };
};
#endif
