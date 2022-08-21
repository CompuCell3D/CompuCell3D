#ifndef POLARIZATION23PATA_H
#define POLARIZATION23PATA_H

#include "Polarization23DLLSpecifier.h"
#include <PublicUtilities/Vector3.h>


namespace CompuCell3D {


    class POLARIZATION23_EXPORT Polarization23Data {
    public:
        Polarization23Data() :
                type1(0),
                type2(0),
                lambda(0.0) {};

        ~Polarization23Data() {};

        Vector3 polarizationVec;
        unsigned char type1, type2;
        double lambda;


    };
};
#endif
