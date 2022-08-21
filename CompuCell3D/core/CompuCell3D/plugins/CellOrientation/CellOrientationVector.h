
#ifndef CELLORIENTATIONVECTOR_H
#define CELLORIENTATIONVECTOR_H

#include "CellOrientationDLLSpecifier.h"

namespace CompuCell3D {

    class CellG;


    class CellOrientationVector {
    public:

        CellOrientationVector(float _x = 0.0, float _y = 0.0, float _z = 0.0)
                : x(_x), y(_y), z(_z) {}

        float x, y, z;

    };

};
#endif


