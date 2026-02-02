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
                xy_size(x_size * y_size)                
        {}

        Field3DIndex(const Dim3D &_dim);

        long index(const Point3D &pt) const {

            return (pt.x + ((pt.y + (pt.z * y_size)) * x_size));
        }

        Point3D index2Point(cc3d_index_t index) const {
            using index_math_t = std::ptrdiff_t;  // signed, pointer-sized

            const index_math_t idx = static_cast<index_math_t>(index);
            const index_math_t xsz = x_size;
            const index_math_t xysz = xy_size;

            index_math_t z_m   = idx / xysz;
            index_math_t rem_m = idx - z_m * xysz;
            index_math_t y_m   = rem_m / xsz;
            index_math_t x_m   = rem_m - y_m * xsz;

#ifndef NDEBUG
            assert(x_m >= 0 && y_m >= 0 && z_m >= 0);
            assert(x_m <= std::numeric_limits<cc3d_dim_t>::max());
            assert(y_m <= std::numeric_limits<cc3d_dim_t>::max());
            assert(z_m <= std::numeric_limits<cc3d_dim_t>::max());
#endif

            return Point3D(
                static_cast<cc3d_dim_t>(x_m),
                static_cast<cc3d_dim_t>(y_m),
                static_cast<cc3d_dim_t>(z_m)
            );
        }

//         Point3D index2Point(cc3d_index_t index) const {
//             cc3d_index_t z_i   = index / xy_size;
//             cc3d_index_t rem_i = index - z_i * xy_size;
//             cc3d_index_t y_i   = rem_i / x_size;
//             cc3d_index_t x_i   = rem_i - y_i * x_size;
//
//             // Optional debug-time safety checks
// #ifndef NDEBUG
//             assert(x_i <= std::numeric_limits<cc3d_dim_t>::max());
//             assert(y_i <= std::numeric_limits<cc3d_dim_t>::max());
//             assert(z_i <= std::numeric_limits<cc3d_dim_t>::max());
// #endif
//
//             return Point3D(
//                 static_cast<cc3d_dim_t>(x_i),
//                 static_cast<cc3d_dim_t>(y_i),
//                 static_cast<cc3d_dim_t>(z_i)
//             );
//         }
        // Point3D index2Point(long _index) const {
        //     cc3d_dim_t x, y, z, rem;
        //     z = _index / xy_size;
        //     rem = _index - z * xy_size;
        //     y = rem / x_size;
        //     x = rem - y * x_size;
        //     return Point3D(x, y, z);
        // }

        long getMaxIndex() const { return maxIndex; }

    private:

        int x_size;
        int y_size;
        int z_size;
        std::ptrdiff_t xy_size;
        std::ptrdiff_t maxIndex=0;

    };

};


#endif
