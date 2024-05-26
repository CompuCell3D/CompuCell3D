#ifndef BOUNDARYTYPEDEFINITIONS_H
#define BOUNDARYTYPEDEFINITIONS_H

#include <string>


namespace CompuCell3D {

    enum LatticeType {
        SQUARE_LATTICE = 1, HEXAGONAL_LATTICE = 2
    };

    // supporting 2.d D simulations - works only in xy plane  - so that we can handle hex lattice and square lattice
    // in the same manner - codewise
    enum DimensionType {
        DIM_DEFAULT = 1, DIM_2_5 = 2
    };

    class LatticeMultiplicativeFactors {
    public:
        LatticeMultiplicativeFactors(double _volumeMF = 1.0, double _surfaceMF = 1.0, double _lengthMF = 1.0) :
                volumeMF(_volumeMF),
                surfaceMF(_surfaceMF),
                lengthMF(_lengthMF) {}

        double volumeMF;
        double surfaceMF;
        double lengthMF;
    };

};

#endif
