#ifndef NEIGHBOR_H
#define NEIGHBOR_H

#include "../Field3D/Point3D.h"
#include <Utils/Coordinates3D.h>
#include <iostream>

namespace CompuCell3D {

    /**
     * Used by NeighborFinder to hold the offset to a neighbor Point3D and
     * it's distance.
     */
    class Neighbor {
    public:
        Point3D pt;
        double distance;
        Coordinates3D<double> ptTrans;

        Neighbor() : distance(0) {}

        Neighbor(const Point3D pt, const double distance,
                 const Coordinates3D<double> _ptTrans = Coordinates3D<double>(.0, .0, .0)) :
                pt(pt), distance(distance), ptTrans(_ptTrans) {}
    };


    inline std::ostream &operator<<(std::ostream &_scr, const Neighbor &_n) {
        using namespace std;
        _scr << "pt=" << _n.pt << " ptTrans=" << _n.ptTrans << " distance=" << _n.distance;
        return _scr;

    }
};
#endif
