#ifndef FIELD3DINDEX_H
#define FIELD3DINDEX_H

#include "Dim3D.h"


namespace CompuCell3D {

    class Dim3D;

    class Point3D;

    class Field3DIndex {
    public:
        Field3DIndex() :
                x_size(0),
                y_size(0),
                z_size(0),
                xy_size(x_size * y_size) {}

        Field3DIndex(const Dim3D &_dim);

        long index(const Point3D &pt) const {

            return (pt.x + ((pt.y + (pt.z * y_size)) * x_size));
        }

        Point3D index2Point(long _index) const {
            short x, y, z, rem;
            z = _index / xy_size;
            rem = _index - z * xy_size;
            y = rem / x_size;
            x = rem - y * x_size;
            return Point3D(x, y, z);
        }

        long getMaxIndex() const { return maxIndex; }

    private:

        int x_size;
        int y_size;
        int z_size;
        int xy_size;
        int maxIndex;

    };

};


#endif
