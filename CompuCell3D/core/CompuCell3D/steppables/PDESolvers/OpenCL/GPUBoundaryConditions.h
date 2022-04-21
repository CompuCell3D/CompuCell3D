
#ifndef GPU_BOUNDARY_CONDITIONS_H
#define GPU_BOUNDARY_CONDITIONS_H

//this file works both on CPU and GPU sides, so changes should be carefuly made

typedef enum {
    BC_PERIODIC,
    BC_CONSTANT_VALUE,
    BC_CONSTANT_DERIVATIVE
} BCType;

typedef enum {
    BC_MIN_X = 0, BC_MAX_X,
    BC_MIN_Y, BC_MAX_Y,
    BC_MIN_Z, BC_MAX_Z
} BCPosition;

//tot use on GPU side 
typedef struct GPUBoundaryConditions {

    BCType planePositions[6];
    float values[6];
} GPUBoundaryConditions_t;

inline bool hasNonPeriodic(GPUBoundaryConditions_t const *bc) {

    for (int bcPos = BC_MIN_X; bcPos <= BC_MAX_Z; ++bcPos) {
        if (bc->planePositions[bcPos] != BC_PERIODIC)
            return true;
    }
    return false;
}

#endif 
