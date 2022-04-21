#ifndef BOUNDARYMONITORPLUGIN_H
#define BOUNDARYMONITORPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "BoundaryMonitorDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {

    class Simulator;

    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class ParallelUtilsOpenMP;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class BOUNDARYMONITOR_EXPORT  BoundaryMonitorPlugin : public Plugin, public CellGChangeWatcher {

    private:

        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
        Array3DCUDA<unsigned char> *boundaryArray;
        unsigned int maxNeighborIndex;


    public:

        BoundaryMonitorPlugin();

        virtual ~BoundaryMonitorPlugin();

        Array3DCUDA<unsigned char> *getBoundaryArray();

        // CellChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);


        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual void handleEvent(CC3DEvent &_event);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif
        
