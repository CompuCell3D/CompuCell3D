#ifndef BOUNDARYCONDITIONSPECIFIER_H
#define BOUNDARYCONDITIONSPECIFIER_H

namespace CompuCell3D {

    struct PDESOLVERS_EXPORT BoundaryConditionSpecifier{
            enum BCType{
                PERIODIC,
                        CONSTANT_VALUE,
                        CONSTANT_DERIVATIVE
            };

            enum BCPosition{
                INTERNAL = -2, BOUNDARY,
                        MIN_X, MAX_X,
                        MIN_Y, MAX_Y,
                        MIN_Z, MAX_Z
            };
            BoundaryConditionSpecifier(){
                planePositions[0] = PERIODIC;//min X
                planePositions[1] = PERIODIC;//max X
                planePositions[2] = PERIODIC;//min Y
                planePositions[3] = PERIODIC;//max Y
                planePositions[4] = PERIODIC;//min Z
                planePositions[5] = PERIODIC;//max Z
            }

            BCType planePositions[6];
            double values[6];
    };

}

#endif