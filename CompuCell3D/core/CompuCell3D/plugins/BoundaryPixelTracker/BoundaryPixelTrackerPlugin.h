#ifndef BOUNDARYPIXELTRACKERPLUGIN_H
#define BOUNDARYPIXELTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "BoundaryPixelTracker.h"
#include "BoundaryPixelTrackerDLLSpecifier.h"


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


    class BOUNDARYPIXELTRACKER_EXPORT BoundaryPixelTrackerPlugin : public Plugin, public CellGChangeWatcher {

        //WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <BoundaryPixelTracker> boundaryPixelTrackerAccessor;
        Simulator *simulator;
        Potts3D *potts;
        unsigned int maxNeighborIndex;
        unsigned int neighborOrder;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;
        std::vector<int> extraBoundariesNeighborOrder;
        std::vector<int> extraBoundariesMaxNeighborIndex;

    public:
        BoundaryPixelTrackerPlugin();

        virtual ~BoundaryPixelTrackerPlugin();


        // Field3DChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);


        void updateBoundaryPixels(const Point3D &pt, CellG *newCell, CellG *oldCell, int indexOfExtraBoundary = -1);


        //Plugin interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *_simulators);

        virtual void handleEvent(CC3DEvent &_event);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        std::set <BoundaryPixelTrackerData> *getPixelSetForNeighborOrderPtr(CellG *_cell, int _neighborOrder);

        ExtraMembersGroupAccessor <BoundaryPixelTracker> *
        getBoundaryPixelTrackerAccessorPtr() { return &boundaryPixelTrackerAccessor; }

        //had to include this function to get set iteration working properly
        // with Python , and Player that has restart capabilities
        BoundaryPixelTrackerData *getBoundaryPixelTrackerData(BoundaryPixelTrackerData *_psd) { return _psd; }


    };
};
#endif
