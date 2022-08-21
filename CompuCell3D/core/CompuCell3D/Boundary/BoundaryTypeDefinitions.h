#ifndef BOUNDARYTYPEDEFINITIONS_H
#define BOUNDARYTYPEDEFINITIONS_H

#include <string>


namespace CompuCell3D {

    enum LatticeType {
        SQUARE_LATTICE = 1, HEXAGONAL_LATTICE = 2
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
