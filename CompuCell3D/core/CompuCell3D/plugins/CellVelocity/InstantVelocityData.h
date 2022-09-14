#ifndef INSTANTVELOCITYDATA_H
#define INSTANTVELOCITYDATA_H

#include <Utils/Coordinates3D.h>

namespace CompuCell3D {

    class InstantCellVelocityData {
    public:
        Coordinates3D<float> oldCellCM;
        Coordinates3D<float> newCellCM;
        Coordinates3D<float> oldCellV;
        Coordinates3D<float> newCellV;

        void zeroAll() {

            Coordinates3D<float> zeroVec(0., 0., 0.);

            oldCellCM = zeroVec;
            newCellCM = zeroVec;
            oldCellV = zeroVec;
            newCellV = zeroVec;

        }


    };


};
#endif
