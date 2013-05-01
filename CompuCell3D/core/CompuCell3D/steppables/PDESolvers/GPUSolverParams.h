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
#define BLOCK_SIZE 8

typedef struct SolverParams{
	unsigned int dimx,dimy,dimz;
	unsigned int numberOfCelltypes;
	float  decayCoef[UCHAR_MAX+1];
	float  diffCoef[UCHAR_MAX+1];
	float  secretionCoef[UCHAR_MAX+1];
	float dx2;
	float dt;

} SolverParams_t;



typedef struct UniSolverParams{//TODO: some of them are field params, not a solver param

//seems that these structures should come first (because of alligment issues?)
#ifdef __OPENCL_VERSION__ 
	int4 nbhdShifts[6];
	//int nbhdShifts[24];
#else 
	cl_int4 nbhdShifts[6];//TODO: this should not be a part of the params
	//int nbhdShifts[24];
#endif

	float  diffCoef[UCHAR_MAX+1];
	float  decayCoef[UCHAR_MAX+1];
	float dx;//cell size
	int hexLattice;//size of bool is not defined in OpenCL

//TODO: update for hexagonal mesh	
	
	int nbhdConcLen;
	int nbhdDiffLen;
	int xDim;
	int yDim;
	int zDim;

} UniSolverParams_t;

inline
unsigned int fieldLength(UniSolverParams_t const *sps){
	assert(sps);
	return sps->xDim*sps->yDim*sps->zDim;
}



#endif