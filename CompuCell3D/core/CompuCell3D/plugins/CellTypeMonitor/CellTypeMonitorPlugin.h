#ifndef CELLTYPEMONITORPLUGIN_H
#define CELLTYPEMONITORPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "CellTypeMonitorDLLSpecifier.h"

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

    class CELLTYPEMONITOR_EXPORT CellTypeMonitorPlugin : public Plugin, public CellGChangeWatcher {

    private:

        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;
        WatchableField3D<CellG *> *cellFieldG;
        Array3DCUDA<unsigned char> *cellTypeArray;

        // this should have been Array3DCUDA<long> but openCL on windows does not "like" longs so I am using float .
        Array3DCUDA<float> *cellIdArray;
        unsigned char mediumType;
    public:

        CellTypeMonitorPlugin();

        virtual ~CellTypeMonitorPlugin();

        virtual Array3DCUDA<unsigned char> *getCellTypeArray() { return cellTypeArray; }

        virtual Array3DCUDA<float> *getCellIdArray() { return cellIdArray; }

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
        
