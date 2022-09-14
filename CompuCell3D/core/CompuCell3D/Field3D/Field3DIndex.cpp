#include "Field3DIndex.h"

using namespace CompuCell3D;

Field3DIndex::Field3DIndex(const Dim3D &_dim) :
        x_size(_dim.x),
        y_size(_dim.y),
        z_size(_dim.z),
        xy_size(_dim.x * _dim.y) {
    Point3D maxPt(x_size - 1, y_size - 1, z_size - 1);
    maxIndex = index(maxPt);
}
