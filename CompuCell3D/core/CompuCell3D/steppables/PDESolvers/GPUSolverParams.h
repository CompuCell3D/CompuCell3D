#ifndef _GPUSOLVERPARAMS_H
#define _GPUSOLVERPARAMS_H

#ifndef __OPENCL_VERSION__

#include <cassert>

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif
#endif

//!!! Do not add any headers here without checking as it may brake the OpenCL kernel(s)

#ifndef UCHAR_MAX
#define UCHAR_MAX 255
#endif

// Thread block size
#define BLOCK_SIZE 16

typedef struct SolverParams {
    unsigned int dimx, dimy, dimz;
    unsigned int numberOfCelltypes;
    float decayCoef[UCHAR_MAX + 1];
    float diffCoef[UCHAR_MAX + 1];

    float dx2;
    float dt;

} SolverParams_t;

//secretionData enums

enum SecretionDataEnum {
    SECRETION_CONST, MAX_UPTAKE, RELATIVE_UPTAKE
};

enum BCType_GPU {
    PERIODIC,
    CONSTANT_VALUE,
    CONSTANT_DERIVATIVE
};

enum BCPosition_GPU {
    INTERNAL = -2, BOUNDARY,
    MIN_X, MAX_X,
    MIN_Y, MAX_Y,
    MIN_Z, MAX_Z
};

//this is repetition of BOundaryConditionSpecifier.h but writen in a way that does not include C++ bits - may clean up that later
typedef struct BCSpecifier {
    //BCType planePositions[6];
    int planePositions[6];
    float values[6];

} BCSpecifier;


typedef struct UniSolverParams {//TODO: some of them are field params, not a solver param

//seems that these structures should come first (because of alligment issues?)
#ifdef __OPENCL_VERSION__
    int4 nbhdShifts[6];
    //int nbhdShifts[24];
#else
    cl_int4 nbhdShifts[6];//TODO: this should not be a part of the params
    //int nbhdShifts[24];
#endif

    float diffCoef[UCHAR_MAX + 1];
    float decayCoef[UCHAR_MAX + 1];
    float secretionData[UCHAR_MAX + 1][3];
    float secretionConstantConcentrationData[UCHAR_MAX + 1];
    float secretionOnContactData[
            UCHAR_MAX + 1][UCHAR_MAX +
                           2]; // secretionOnContactData[cell_type][UCHAR_MAX+1] is a flag that says whether cell_type has any secrete on contact data or not
    int secretionDoUptake;
    float dx;//cell size
    float dt;//cell size
    int hexLattice;//size of bool is not defined in OpenCL

//TODO: update for hexagonal mesh	

    int nbhdConcLen;
    int nbhdDiffLen;
    int xDim;
    int yDim;
    int zDim;
    int extraTimesPerMCS;

} UniSolverParams_t;

inline
unsigned int fieldLength(UniSolverParams_t const *sps) {
    assert(sps);
    return sps->xDim * sps->yDim * sps->zDim;
}


#endif