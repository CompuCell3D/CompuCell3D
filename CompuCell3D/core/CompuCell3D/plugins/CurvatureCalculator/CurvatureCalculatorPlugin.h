
#ifndef CURVATURECALCULATORPLUGIN_H
#define CURVATURECALCULATORPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "CurvatureCalculatorDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Simulator;

    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class ParallelUtilsOpenMP;

    class BoundaryPixelTrackerPlugin;

    class NeighborTrackerPlugin;

    class NeighborTracker;

    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class CURVATURECALCULATOR_EXPORT  CurvatureCalculatorPlugin : public Plugin, public CellGChangeWatcher {

    private:

        CC3DXMLElement *xmlData;

        Potts3D *potts;

        Simulator *sim;

        ParallelUtilsOpenMP *pUtils;

        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;

        WatchableField3D<CellG *> *cellFieldG;

        ExtraMembersGroupAccessor <NeighborTracker> *neighborTrackerAccessorPtr;

        BoundaryPixelTrackerPlugin *boundary_pixel_tracker_plugin;

        int maxNeighborIndex;
        int neighborOrderProbCalc;
        int maxNeighborIndexProbCalc;

    public:

        CurvatureCalculatorPlugin();

        virtual ~CurvatureCalculatorPlugin();

        // utility functions
        std::map<long, float> getProbabilityByNeighbor(CellG *cell, float J, float T);

        float getGrowthProbability(const Point3D &neighborPt, CellG *neighborCell, CellG *cell, float J, float T);

        // CellChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif

