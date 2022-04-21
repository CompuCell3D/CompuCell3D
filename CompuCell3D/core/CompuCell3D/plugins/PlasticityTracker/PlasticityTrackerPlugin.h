#ifndef REALPLASTICITYTRACKERPLUGIN_H
#define REALPLASTICITYTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "PlasticityTracker.h"
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>


#include "PlasticityTrackerDLLSpecifier.h"

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

    class PLASTICITYTRACKER_EXPORT PlasticityTrackerPlugin : public Plugin, public CellGChangeWatcher {
        ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
        WatchableField3D<CellG *> *cellFieldG;
        Dim3D fieldDim;
        ExtraMembersGroupAccessor <PlasticityTracker> plasticityTrackerAccessor;
        ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr;

        Simulator *simulator;
        CellInventory *cellInventoryPtr;
        bool initialized;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;
    public:
        PlasticityTrackerPlugin();

        virtual ~PlasticityTrackerPlugin();

        // SimObject interface
        virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        // BCGChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        ExtraMembersGroupAccessor <PlasticityTracker> *
        getPlasticityTrackerAccessorPtr() { return &plasticityTrackerAccessor; }

        //had to include this function to get set itereation working properly
        // with Python , and Player that has restart capabilities
        PlasticityTrackerData *getPlasticityTrackerData(PlasticityTrackerData *_psd) { return _psd; }

        void initializePlasticityNeighborList();

        void addPlasticityNeighborList();

    protected:


        std::set <std::string> plasticityTypesNames;
        std::set<unsigned char> plasticityTypes;

    };
};
#endif
