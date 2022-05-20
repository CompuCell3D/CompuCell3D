#ifndef NEIGHBORTRACKER_H
#define NEIGHBORTRACKER_H


/**
@author m
*/
#include <set>

#include "NeighborTrackerDLLSpecifier.h"


namespace CompuCell3D {

    class CellG;

    //common surface area is expressed in unitsa of elementary surfaces not actual physical units. If necessary it may
    //need to be transformed to physical units by multiplying it by surface latticemultiplicative factor
    class NEIGHBORTRACKER_EXPORT NeighborSurfaceData {
    public:

        NeighborSurfaceData(CellG *_neighborAddress = 0, int _commonSurfaceArea = 0)
                : neighborAddress(_neighborAddress),
                  commonSurfaceArea(_commonSurfaceArea) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const NeighborSurfaceData &_rhs) const {
            return neighborAddress < _rhs.neighborAddress;
        }

        ///had to do this dirty trick to work around a problem that iterators of a set give access in read-only mode
        ///Note : You should NEVER change this way class members that are used in <operator this will corrupt set container
        ///CAUTION: DO NOT TRY TO MODIFY pixelIndex - you will corrupt set container
        void incrementCommonSurfaceArea(const NeighborSurfaceData &_neighborSurfaceData) const {
            ++((const_cast<NeighborSurfaceData &>(_neighborSurfaceData)).commonSurfaceArea);
        }

        ///Note : You should NEVER change this way class members that are used in <operator this will corrupt set container
        void decrementCommonSurfaceArea(const NeighborSurfaceData &_neighborSurfaceData) const {
            --((const_cast<NeighborSurfaceData &>(_neighborSurfaceData)).commonSurfaceArea);
        }

        bool OKToRemove() const { return commonSurfaceArea == 0; }

        bool operator==(const NeighborSurfaceData &_rhs) const {
            return (neighborAddress == _rhs.neighborAddress) && (commonSurfaceArea == _rhs.commonSurfaceArea);
        }

        ///members
        int getCommonSurfaceArea() { return commonSurfaceArea; }

        CellG *neighborAddress;
        int commonSurfaceArea;


    };


    class NEIGHBORTRACKER_EXPORT NeighborTracker {
    public:
        NeighborTracker() {};

        int trackerNumber() { return 1234321; }

        ~NeighborTracker() {};
        //stores ptrs to cell neighbors i.e. each cell keeps track of its neighbors
        std::set <NeighborSurfaceData> cellNeighbors;

    };
};
#endif
