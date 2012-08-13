#ifndef _GPUSOLVERPARAMS_H
#define _GPUSOLVERPARAMS_H

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
	float dx;
	float dt;

} SolverParams_t;

typedef struct UniSolverParams{
	float  diffCoef[UCHAR_MAX+1];
	float  decayCoef[UCHAR_MAX+1];
	float dx;
	float dt;
	int hexLattice;//size of bool is not defined in OpenCL

	int nbhdConcLen;
	int nbhdDiffLen;
	int xDim;
	int yDim;
	int zDim;

} UniSolverParams_t;



#endif