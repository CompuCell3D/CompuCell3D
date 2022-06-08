#ifndef BOUNDARYTRACKER_H
#define BOUNDARYTRACKER_H


/**
@author m
*/
#include <set>

namespace CompuCell3D {

    class CellG;

    class BoundaryData {
    public:

        BoundaryData(long _pixelIndex = 0, short _numberOfForeignNeighbors = 0)
                : pixelIndex(_pixelIndex),
                  numberOfForeignNeighbors(_numberOfForeignNeighbors) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const BoundaryData &_rhs) const {
            return pixelIndex < _rhs.pixelIndex;
        }

        bool operator==(const BoundaryData &_rhs) const {
            return pixelIndex == _rhs.pixelIndex;
        }


        ///had to do this dirty trick to work around a problem that iterators of a set give access in read-only mode
        ///Note : You should NEVER change this way class members that are used in <operator this will corrupt set container
        ///CAUTION: DO NOT TRY TO MODIFY pixelIndex - you will corrupt set container
        void incrementNumberOfForeignNeighbors(const BoundaryData &_boundaryData) const {
            ++((const_cast<BoundaryData &>(_boundaryData)).numberOfForeignNeighbors);
        }

        ///Note : You should NEVER change this way class members that are used in <operator this will corrupt set container
        void decrementNumberOfForeignNeighbors(const BoundaryData &_boundaryData) const {
            --((const_cast<BoundaryData &>(_boundaryData)).numberOfForeignNeighbors);
        }

        bool OKToRemove() const { return numberOfForeignNeighbors <= 0; }

        ///members
        long pixelIndex;
        short numberOfForeignNeighbors;


    };

    class NeighborSurfaceData {
    public:

        NeighborSurfaceData(CellG *_neighborAddress = 0, short _commonSurfaceArea = 0)
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

        ///members

        CellG *neighborAddress;
        short commonSurfaceArea;


    };


    class BoundaryTracker {
    public:
        CellBoundaryTracker() {};

        ~CellBoundaryTracker() {};
        //perhaps it will have to be changed to set<long>
        std::set <BoundaryData> boundary; //stores cell boundary.
        // Each Point3D has an index associated with it - it is simply an offset
        // in the Field3D array

    };
};
#endif
