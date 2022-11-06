#include "AdjacentNeighbor.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
AdjacentNeighbor::AdjacentNeighbor(const Dim3D &_dim) :
        periodicX(false),
        periodicY(false),
        periodicZ(false) {
    ///here I will initialize vector of offsets
    initialize(_dim);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdjacentNeighbor::initialize(const Dim3D &_dim) {

    depth = 1;
    fieldDim = _dim;
    field3DIndex = Field3DIndex(_dim);

    adjNeighborOffsetsInner.assign((2 * depth + 1) * (2 * depth + 1) * (2 * depth + 1) - 1,
                                   Point3D(0, 0, 0));   //will not include 0 in the offset table -
    // that's why I subract 1 from vector dimension

    adjFace2FaceNeighborOffsetsInner.assign(6, Point3D(0, 0, 0));

    //remove it later - testing now
    adjNeighborOffsets.assign((2 * depth + 1) * (2 * depth + 1) * (2 * depth + 1) - 1, 0);
    adjFace2FaceNeighborOffsets.assign(6, 0);

    long index;
    long counter = 0;
    Point3D self(0, 0, 0);
    for (short x = -depth; x <= depth; ++x)
        for (short y = -depth; y <= depth; ++y)
            for (short z = -depth; z <= depth; ++z) {
                Point3D pt(x, y, z);
                index = field3DIndex.index(pt);
                if (!(self == pt)) {
                    adjNeighborOffsetsInner[counter] = pt;
                    adjNeighborOffsets[counter] = index;
                    ++counter;
                }

            }

    /// initializing face2face offsets
    counter = 0;
    for (short x = -1; x <= 1; ++x)
        for (short y = -1; y <= 1; ++y)
            for (short z = -1; z <= 1; ++z) {
                Point3D pt(x, y, z);

                index = field3DIndex.index(pt);

                if (!(self == pt) && !(distance(x, y, z) > 1.0)) {
                    adjFace2FaceNeighborOffsetsInner[counter] = pt;
                    adjFace2FaceNeighborOffsets[counter] = index;
                    ++counter;
                }

            }

    adjNeighborOffsetsBoundary.assign(adjNeighborOffsetsInner.begin(), adjNeighborOffsetsInner.end());
    adjFace2FaceNeighborOffsetsBoundary.assign(adjFace2FaceNeighborOffsetsInner.begin(),
                                               adjFace2FaceNeighborOffsetsInner.end());

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double AdjacentNeighbor::distance(double x, double y, double z) {
    return sqrt(x * x + y * y + z * z);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdjacentNeighbor::setPeriodicX() {
    if (periodicX)
        return;     ///do nothing if someone has already called this function

    periodicX = true;
    short maxXPlus = fieldDim.x - 1;
    short maxXMinus = -(fieldDim.x - 1);
    Point3D self(0, 0, 0);
    for (short y = -1; y <= 1; ++y)
        for (short z = -1; z <= 1; ++z) {
            Point3D ptPlus(maxXPlus, y, z);
            Point3D ptMinus(maxXMinus, y, z);
            if (!(ptPlus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptPlus);
            }
            if (!(ptMinus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptMinus);
            }

        }

    ///adding Face2FaceNeighborOffsets
    Point3D ptPlus(maxXPlus, 0, 0);
    Point3D ptMinus(maxXMinus, 0, 0);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptPlus);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptMinus);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdjacentNeighbor::setPeriodicY() {

    if (periodicY)
        return;     ///do nothing if someone has already called this function

    periodicY = true;
    short maxYPlus = fieldDim.y - 1;
    short maxYMinus = -(fieldDim.y - 1);
    Point3D self(0, 0, 0);
    for (short x = -1; x <= 1; ++x)
        for (short z = -1; z <= 1; ++z) {
            Point3D ptPlus(x, maxYPlus, z);
            Point3D ptMinus(x, maxYMinus, z);
            if (!(ptPlus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptPlus);
            }
            if (!(ptMinus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptMinus);
            }

        }

    Point3D ptPlus(0, maxYPlus, 0);
    Point3D ptMinus(0, maxYMinus, 0);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptPlus);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptMinus);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdjacentNeighbor::setPeriodicZ() {
    if (periodicZ)
        return;     ///do nothing if someone has already called this function

    periodicZ = true;

    short maxZPlus = fieldDim.z - 1;
    short maxZMinus = -(fieldDim.z - 1);
    Point3D self(0, 0, 0);
    for (short x = -1; x <= 1; ++x)
        for (short y = -1; y <= 1; ++y) {
            Point3D ptPlus(x, y, maxZPlus);
            Point3D ptMinus(x, y, maxZMinus);
            if (!(ptPlus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptPlus);
            }
            if (!(ptMinus == self)) {
                adjNeighborOffsetsBoundary.push_back(ptMinus);
            }

        }
    CC3D_Log(LOG_DEBUG) << "adjNeighborOffsetsBoundary.size()="<<adjNeighborOffsetsBoundary.size();
    Point3D ptPlus(0, 0, maxZPlus);
    Point3D ptMinus(0, 0, maxZMinus);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptPlus);
    adjFace2FaceNeighborOffsetsBoundary.push_back(ptMinus);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
AdjacentNeighbor::~AdjacentNeighbor() {
}


