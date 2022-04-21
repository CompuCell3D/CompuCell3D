#ifndef BOUNDARYPIXELTRACKER_H
#define BOUNDARYPIXELTRACKER_H

/**
@author m
*/
#include <CompuCell3D/CC3D.h>
#include "BoundaryPixelTrackerDLLSpecifier.h"

namespace CompuCell3D {

    //common surface area is expressed in units of elementary surfaces not actual physical units. If necessary it may
    //need to be transformed to physical units by multiplying it by surface lattice multiplicative factor
    class BOUNDARYPIXELTRACKER_EXPORT BoundaryPixelTrackerData {
    public:
        BoundaryPixelTrackerData() {
            pixel = Point3D();
        }

        BoundaryPixelTrackerData(Point3D _pixel)
                : pixel(_pixel) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const BoundaryPixelTrackerData &_rhs) const {
            return pixel.x < _rhs.pixel.x || (!(_rhs.pixel.x < pixel.x) && pixel.y < _rhs.pixel.y)
                   || (!(_rhs.pixel.x < pixel.x) && !(_rhs.pixel.y < pixel.y) && pixel.z < _rhs.pixel.z);
        }

        bool operator==(const BoundaryPixelTrackerData &_rhs) const {
            return pixel == _rhs.pixel;
        }

        // members
        Point3D pixel;

    };


    class BOUNDARYPIXELTRACKER_EXPORT BoundaryPixelTracker {
    public:
        BoundaryPixelTracker() {};

        ~BoundaryPixelTracker() {};
        std::set <BoundaryPixelTrackerData> pixelSet; //stores pixels belonging to a given cell

        // {search_depth: boundary pixelSet} - the map is indexed by the NeighborOrder
        std::map<int, std::set<BoundaryPixelTrackerData> > pixelSetMap;


    };
};
#endif
