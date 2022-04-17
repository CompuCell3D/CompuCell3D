#ifndef ADJACENTNEIGHBOR_H
#define ADJACENTNEIGHBOR_H

#include "Point3D.h"
#include "Field3DIndex.h"
#include <vector>

/**
@author m
*/
/// This class will cache offsets of a given lattice site that will lead to adjacent lattice points
/// you will have to use getByOffset function from Field3D class to access lattice points
/// Notice that offsets may be negative as well. getByIndex from Field3DImpl.h will return
/// default value of the field element if the array index is out range (i.e negative or greater than internal field array len
/// see Field3DImpl.h for the details).
/// Example: let's say a given point has field array index 20. To access its adjacent neighbors
/// you add 20+neighbor_offset and call getByIndex(20+neighbor_offset)
/// adjFace2FaceNeighborOffsets stores offsets of a neighboring cells that touch a given cell face to face and not just by corner or edge

namespace CompuCell3D {

    class AdjacentNeighbor {
    public:
        explicit AdjacentNeighbor() :
                periodicX(false),
                periodicY(false),
                periodicZ(false) {}

        explicit AdjacentNeighbor(const Dim3D &_dim);

        ~AdjacentNeighbor();

        void initialize(const Dim3D &_dim);

        std::vector<long> const &getAdjNeighborOffsetVec() const { return adjNeighborOffsets; }

        std::vector<long> const &getAdjFace2FaceNeighborOffsetVec() const { return adjFace2FaceNeighborOffsets; }

        std::vector <Point3D> const &getAdjNeighborOffsetVec(const Point3D &_pt) const {

            if (isInner(_pt)) {
                return adjNeighborOffsetsInner;
            } else {
                return adjNeighborOffsetsBoundary;
            }

        }

        std::vector <Point3D> const &getAdjFace2FaceNeighborOffsetVec(const Point3D &_pt) const {
            //have to correct this function  - as of now it gives segfault

            if (isInner(_pt)) {
                return adjFace2FaceNeighborOffsetsInner;
            } else {
                return adjFace2FaceNeighborOffsetsBoundary;
            }

        }

        Dim3D getFieldDim() const { return fieldDim; }

        Field3DIndex const &getField3DIndex() const { return field3DIndex; }

        double distance(double, double, double);

        void setPeriodicX();

        void setPeriodicY();

        void setPeriodicZ();

        bool isInner(const Point3D &_pt) const {
            return (_pt.x > 0 && _pt.x < (fieldDim.x - 1)
                    && _pt.y > 0 && _pt.y < (fieldDim.y - 1)
                    && _pt.z > 0 && _pt.z < (fieldDim.z - 1)
            );
        }

    protected:

        std::vector<long> adjNeighborOffsets;
        std::vector<long> adjFace2FaceNeighborOffsets;
        std::vector <Point3D> adjNeighborOffsetsInner;
        std::vector <Point3D> adjNeighborOffsetsBoundary;
        std::vector <Point3D> adjFace2FaceNeighborOffsetsInner;
        std::vector <Point3D> adjFace2FaceNeighborOffsetsBoundary;
        bool periodicX;
        bool periodicY;
        bool periodicZ;
        Field3DIndex field3DIndex;
        int depth;
        Dim3D fieldDim;
    };
};
#endif
