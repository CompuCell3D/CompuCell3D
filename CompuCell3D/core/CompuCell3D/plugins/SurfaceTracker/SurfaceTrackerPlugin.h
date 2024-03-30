#ifndef SURFACETRACKERPLUGIN_H
#define SURFACETRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "SurfaceTrackerDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {


    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;

    class BoundaryStrategy;

    class Potts3D;

    class SURFACETRACKER_EXPORT SurfaceTrackerPlugin : public Plugin, public CellGChangeWatcher {


        WatchableField3D<CellG *> *cellFieldG;
        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;
        LatticeMultiplicativeFactors lmf;
        Potts3D *potts;

    public:

        SurfaceTrackerPlugin();

        virtual ~SurfaceTrackerPlugin();


        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        const LatticeMultiplicativeFactors &getLatticeMultiplicativeFactors() const { return lmf; }

        unsigned int getMaxNeighborIndex() { return maxNeighborIndex; }
        virtual void setNeighborOrder(unsigned int);

        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);


        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif
