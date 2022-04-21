#ifndef PIXELTRACKER_H
#define PIXELTRACKER_H


/**
@author m
*/
#include <set>
#include <CompuCell3D/Field3D/Point3D.h>
#include "PixelTrackerDLLSpecifier.h"

namespace CompuCell3D {


    //common surface area is expressed in unitsa of elementary surfaces not actual physical units. If necessary it may
    //need to be transformed to physical units by multiplying it by surface latticemultiplicative factor
    class PIXELTRACKER_EXPORT PixelTrackerData {
    public:
        PixelTrackerData() {
            pixel = Point3D();
        }

        PixelTrackerData(Point3D _pixel)
                : pixel(_pixel) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const PixelTrackerData &_rhs) const {
            return pixel.x < _rhs.pixel.x || (!(_rhs.pixel.x < pixel.x) && pixel.y < _rhs.pixel.y)
                   || (!(_rhs.pixel.x < pixel.x) && !(_rhs.pixel.y < pixel.y) && pixel.z < _rhs.pixel.z);
        }

        bool operator==(const PixelTrackerData &_rhs) const {
            return pixel == _rhs.pixel;
        }

        ///members
        Point3D pixel;


    };


    class PIXELTRACKER_EXPORT PixelTracker {
    public:
        PixelTracker() {};

        ~PixelTracker() {};
        std::set <PixelTrackerData> pixelSet; //stores pixels belonging to a given cell

    };
};
#endif
