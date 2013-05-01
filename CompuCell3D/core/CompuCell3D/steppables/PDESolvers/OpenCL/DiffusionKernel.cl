//#include "lib/CompuCell3DSteppables/OpenCL/GPUSolverParams.h"

//Ivan Komarov

//old version with int3 type used produced unexpected behavior, so the type was changed to int4 (April 2012)

void fn(__global float* scratch, int inRangeId, size_t offset, float fill, int DIMX, int DIMY)
{
	scratch[offset+inRangeId]=fill;//body

	if(get_global_id(0)>=DIMX-2)
		scratch[offset+inRangeId+2]=fill;//column

	if(get_global_id(1)>=DIMY-2){
		scratch[offset+inRangeId+2*(DIMX+2)]=fill;//row
		if(get_global_id(1)>=DIMY-2){
			scratch[offset+inRangeId+2*(DIMX+2)+2]=fill;//corner
		}	
	}
}

//for testing purposes
__kernel void foo(__global float* scratch, __global SolverParams_t  const *solverParams)
{
	int DIMX=solverParams->dimx;
	int DIMY=solverParams->dimy;
	int DIMZ=solverParams->dimz;

	/*if(get_global_id(0)==0&&get_global_id(1)==0){
		for(int i=0; i<(DIMX+2)*(DIMY+2)*(DIMZ+2);++i){
			scratch[i]=1.f;
		}
		for(int z=0; z<DIMZ+2; ++z){
			for(int y=0; y<DIMY+2; ++y){
				for(int x=0; x<DIMX+2; ++x){
					int ind=(DIMX+2)*(DIMY+2)*z+(DIMX+2)*y+x;
						scratch[ind]=2.f;
				}
			}

		}
	}*/
	size_t layerSize=(DIMX+2)*(DIMY+2);
	size_t numSlices=DIMZ/get_global_size(2);
	size_t sliceSize=layerSize*get_global_size(2);

	int inRangeId=get_global_id(2)*layerSize+get_global_id(1)*(DIMX+2)+get_global_id(0);
	//int wgOffset=get_global_id(2)*layerSize;
	float to_fill=get_global_size(2);
	//if(get_global_id(2)==0)
	for (int sn=0; sn<numSlices; ++sn){
		int sliceOffset=sliceSize*sn;
		
		fn(scratch, inRangeId, sliceOffset, to_fill, DIMX, DIMY);
	}

	if(get_global_id(2)<2){//top two layers
		int offsetTillHere=sliceSize*numSlices;

		fn(scratch, inRangeId, offsetTillHere, to_fill, DIMX, DIMY);		
	}
	
}


//TODO: check it all indices computation mess can be wrapped into a structure/class

//TODO: check if the grid dimension can be obtained directly without performance penalty
//instead of passing it into the function
inline
size_t glob3DIndToLinear(int dimX, int dimY, 
	int x, int y, int z)
{
	return z*(dimX+2)*(dimY+2)+y*(dimX+2)+x;
}

inline
size_t block3DIndToLinear(int blockDimX, int blockDimY, 
	int x, int y, int z)
{
	return z*blockDimX*blockDimY+y*blockDimX+x;
}


//obsolete version
__kernel void diff3D(__global const float* field,
    __global const unsigned char * celltype,
    __global float* scratch,
    __global SolverParams_t  const *solverParams,
	__local float *fieldBlock,
    __local unsigned char *celltypeBlock,
    __local float *scratchBlock)
{
    //int bx = blockIdx.x;
    int bx=get_group_id(0);
    //int by = blockIdx.y;
    int by=get_group_id(1);
	int bz=get_group_id(2);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int tz = get_local_id(2);

	
    int DIMX=solverParams->dimx;
    int DIMY=solverParams->dimy;
    int DIMZ=solverParams->dimz;


	int const blockSizeX=get_local_size(0);
	int const blockSizeY=get_local_size(1);
	int const blockSizeZ=get_local_size(2);

	
	int x= bx*blockSizeX+tx;
    int y= by*blockSizeY+ty;
	int z= bz*blockSizeZ+tz;

	//in-block indices
	int bScratchInd=block3DIndToLinear(blockSizeX, blockSizeY, tx, ty, tz);//current index in an unextanded block
	int bExInd     =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+1, tz+1);//current index in an extanded block
	int bExIndXm1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx,   ty+1, tz+1);//neighbour ind at x-1 position
	int bExIndXp1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+2, ty+1, tz+1);//neighbour ind at x+1 position
	int bExIndYm1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty,   tz+1);//neighbour ind at y-1 position
	int bExIndYp1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+2, tz+1);//neighbour ind at y+1 position
	int bExIndZm1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+1, tz  );//neighbour ind at z-1 position
	int bExIndZp1  =block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+1, tz+2);//neighbour ind at z+1 position
	
    //mapping from block,threadIdx to x,y,z of the inner frame
    //might be refactored out to a device function
	{
		{
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+1, y+1, z+1);//might be more effitient to declare this variable out of local scope
			fieldBlock[bExInd] = field[globInd];
			celltypeBlock[bExInd] = celltype[globInd];
		}

		scratchBlock[bScratchInd]=0.0f;

		if (tx==0){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x, y+1, z+1);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, 0, ty+1, tz+1);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];
		}

		if (tx==blockSizeX-1){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+2, y+1, z+1);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, blockSizeX+1, ty+1, tz+1);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];
		}

		if (ty==0){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+1, y, z+1);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, 0, tz+1);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];
		}

		if (ty==blockSizeY-1){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+1, y+2, z+1);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, blockSizeY+1, tz+1);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];
		}

		if (tz==0){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+1, y+1, z);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+1, 0);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];

		}

		if (tz==blockSizeZ-1){
			int globInd=glob3DIndToLinear(DIMX, DIMY, x+1, y+1, z+2);
			int blockInd=block3DIndToLinear(blockSizeX+2, blockSizeY+2, tx+1, ty+1, blockSizeZ+1);
			fieldBlock[blockInd]=field[globInd];
			celltypeBlock[blockInd]=celltype[globInd];
		}
	}
		
    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

    //solve actual diff equation
    float concentrationSum =0.0f;
	float dt_dx2=solverParams->dt/solverParams->dx2;

    int curentCelltype=celltypeBlock[bExInd];

    concentrationSum=fieldBlock[bExIndXp1]+fieldBlock[bExIndYp1]+fieldBlock[bExIndZp1]
		+fieldBlock[bExIndXm1]+fieldBlock[bExIndYm1]+fieldBlock[bExIndZm1]-6*fieldBlock[bExInd];

    __global float const * diffCoef=solverParams->diffCoef;
    __global float const * decayCoef=solverParams->decayCoef;

	concentrationSum*=diffCoef[curentCelltype];
		
    float varDiffSumTerm=0.0f;

    //mixing central difference first derivatives with forward second derivatives does not work
    //terms due to variable diffusion coef
    ////x partial derivatives
    //varDiffSumTerm+=(diffCoef[celltypeBlock[tx+2][ty+1][tz+1]]-diffCoef[celltypeBlock[tx][ty+1][tz+1]])*(fieldBlock[tx+2][ty+1][tz+1]-fieldBlock[tx][ty+1][tz+1]);
    ////y partial derivatives
    //varDiffSumTerm+=(diffCoef[celltypeBlock[tx+1][ty+2][tz+1]]-diffCoef[celltypeBlock[tx+1][ty][tz+1]])*(fieldBlock[tx+1][ty+2][tz+1]-fieldBlock[tx+1][ty][tz+1]);
    ////z partial derivatives
    //varDiffSumTerm+=(diffCoef[celltypeBlock[tx+1][ty+1][tz+2]]-diffCoef[celltypeBlock[tx+1][ty+1][tz]])*(fieldBlock[tx+1][ty+1][tz+2]-fieldBlock[tx+1][ty+1][tz]);

    //scratchBlock[tx][ty][tz]=diffConst*(concentrationSum-6*fieldBlock[tx+1][ty+1][tz+1])+fieldBlock[tx+1][ty+1][tz+1];

    //scratchBlock[tx][ty][tz]=dt_4dx2*(concentrationSum+4*varDiffSumTerm)+fieldBlock[tx+1][ty+1][tz+1];


    //scratchBlock[tx][ty][tz]=dt_4dx2*(concentrationSum+varDiffSumTerm)+fieldBlock[tx+1][ty+1][tz+1];


    //using forward first derivatives
    //x partial derivatives
    varDiffSumTerm+=(diffCoef[celltypeBlock[bExIndXp1]]-diffCoef[curentCelltype])*(fieldBlock[bExIndXp1]-fieldBlock[bExInd]);
    //y partial derivatives
    varDiffSumTerm+=(diffCoef[celltypeBlock[bExIndYp1]]-diffCoef[curentCelltype])*(fieldBlock[bExIndYp1]-fieldBlock[bExInd]);
    //z partial derivatives
    varDiffSumTerm+=(diffCoef[celltypeBlock[bExIndZp1]]-diffCoef[curentCelltype])*(fieldBlock[bExIndZp1]-fieldBlock[bExInd]);


    //OK
    scratchBlock[bScratchInd]=dt_dx2*(concentrationSum+varDiffSumTerm)+(1-solverParams->dt*decayCoef[curentCelltype])*fieldBlock[bExInd];



    //simple consistency check
    //scratchBlock[tx][ty][tz]=concentrationSum;
    //scratchBlock[tx][ty][tz]=fieldBlock[tx+2][ty+1][tz+1]+fieldBlock[tx][ty+1][tz+1]+fieldBlock[tx+1][ty+2][tz+1]+fieldBlock[tx+1][ty][tz+1]+fieldBlock[tx+1][ty+1][tz+2]+fieldBlock[tx+1][ty+1][tz];

    //scratchBlock[tx][ty][tz]=fieldBlock[tx+1][ty+1][tz+1];

    //fieldBlock[tx+1][ty+1][tz+1]=3000.0f;
    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

    //copy scratchBlock to scratch field on the device

    scratch[glob3DIndToLinear(DIMX, DIMY, x+1, y+1, z+1)]=scratchBlock[bScratchInd];
    //scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1]=3000.0;

    //__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE);

    //boundary condition
    //if(x==0){
    //    scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

    //if(x==solverParams->dimx-1){
    //    scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+2]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

    //if(y==0){
    //    scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y)*(DIMX+2)+x+1]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

    //if(y==solverParams->dimy-1){
    //    scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+2)*(DIMX+2)+x+1]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

    //if(z==0){
    //    scratch[(z)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

    //if(z==solverParams->dimz-1){
    //    scratch[(z+2)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1]=scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1];
    //}

}

inline
size_t ext3DIndToLinear(int4 dim, int4 id)
{
	return id.z*(dim.x+2)*(dim.y+2)+id.y*(dim.x+2)+id.x;
}


void globToLocal(int4 g_dim, int4 g_ind, int4 l_dim, int4 l_ind, 
	__global const float* g_field, __global const unsigned char* g_cellType, 
	__local float* l_field, __local unsigned char* l_cellType)
{
	int4 minShift={-1,-1,-1,0};
	int4 maxShift={1,1,1,0};


	//all but boundary work items (threads) copy only one element. 
	//Boundary work items copy an extra layer (usually one more element in one direction)
	if(l_ind.x+minShift.x>0)
		minShift.x=0;

	if(l_ind.y+minShift.y>0)
		minShift.y=0;

	if(l_ind.z+minShift.z>0)
		minShift.z=0;

	if(l_ind.x+maxShift.x<=l_dim.x)
		maxShift.x=0;

	if(l_ind.y+maxShift.y<=l_dim.y)
		maxShift.y=0;

	if(l_ind.z+maxShift.z<=l_dim.z)
		maxShift.z=0;
	
	for(int shiftZ=minShift.z; shiftZ<=maxShift.z; ++shiftZ){
		for(int shiftY=minShift.y; shiftY<=maxShift.y; ++shiftY){
			for(int shiftX=minShift.x; shiftX<=maxShift.x; ++shiftX){
				int l_linearInd=ext3DIndToLinear(l_dim, l_ind+(int4)(shiftX, shiftY, shiftZ,0));
				int g_linearInd=ext3DIndToLinear(g_dim, g_ind+(int4)(shiftX, shiftY, shiftZ,0));
				l_field[l_linearInd]=g_field[g_linearInd];
				l_cellType[l_linearInd]=g_cellType[g_linearInd];
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

void globToLocalEx1(int4 g_dim, int4 g_ind, int4 l_dim, int4 l_ind, 
	__global const float* g_field, __global const unsigned char* g_cellType, __global const float* g_fieldDelta,
	__local float* l_field, __local unsigned char* l_cellType, __local float* l_fieldDelta)
{
	int4 minShift={-1,-1,-1,0};
	int4 maxShift={1,1,1,0};


	//all but boundary work items (threads) copy only one element. 
	//Boundary work items copy an extra layer (usually one more element in one direction)
	if(l_ind.x+minShift.x>0)
		minShift.x=0;

	if(l_ind.y+minShift.y>0)
		minShift.y=0;

	if(l_ind.z+minShift.z>0)
		minShift.z=0;

	if(l_ind.x+maxShift.x<=l_dim.x)
		maxShift.x=0;

	if(l_ind.y+maxShift.y<=l_dim.y)
		maxShift.y=0;

	if(l_ind.z+maxShift.z<=l_dim.z)
		maxShift.z=0;
	
	for(int shiftZ=minShift.z; shiftZ<=maxShift.z; ++shiftZ){
		for(int shiftY=minShift.y; shiftY<=maxShift.y; ++shiftY){
			for(int shiftX=minShift.x; shiftX<=maxShift.x; ++shiftX){
				int l_linearInd=ext3DIndToLinear(l_dim, l_ind+(int4)(shiftX, shiftY, shiftZ,0));
				int g_linearInd=ext3DIndToLinear(g_dim, g_ind+(int4)(shiftX, shiftY, shiftZ,0));
				l_field[l_linearInd]=g_field[g_linearInd];
				l_cellType[l_linearInd]=g_cellType[g_linearInd];
				l_fieldDelta[l_linearInd]=g_fieldDelta[g_linearInd];
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

int4 getShift(int4 pt, int ind, bool isHexLattice, __constant const int4 *nbhdConcEff, int offsetLen)
{
	if(!isHexLattice)
		return nbhdConcEff[ind];
	else
	{
		int row=(pt.z%3)*2+pt.y%2;
		
		return nbhdConcEff[row*offsetLen+ind];
	}
}


__kernel void uniDiff(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType,
	float dt
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)

	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];
	float concentrationSum=0.f;
	if(solverParams->hexLattice){
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
			int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			concentrationSum+=l_field[lInd];
		}
		concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
	}else{
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=nbhdConcShifts[i];
			int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			concentrationSum+=l_field[lInd];
		}
		concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
	}
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;
	if(solverParams->hexLattice){
		for(int i=0; i<solverParams->nbhdDiffLen; ++i){
			int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, solverParams->nbhdDiffLen);
			int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
		}
	}else{
		for(int i=0; i<solverParams->nbhdDiffLen; ++i){
			int lInd=ext3DIndToLinear(l_dim, l_ind+nbhdDiffShifts[i]);
			varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
		}
	}
	

	float dx2=solverParams->dx*solverParams->dx;
	float dt_dx2=dt/dx2;

    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	g_scratch[g_linearInd]=scratch;
	
}


//TODO: fix this by using diffop and taking dt/dx into account
//used in linear solver
__kernel void prod(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType,
	float dt
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)

	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];
	float concentrationSum=0.f;
	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];

	float scratch=0.f;
	if(solverParams->hexLattice){
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
			int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			concentrationSum+=l_field[lInd];
		}
		concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
	}else
	{
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=nbhdConcShifts[i];
			int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			scratch-=l_field[lInd]*currentDiffCoef;
			//concentrationSum-=l_field[lInd];
		}
		//concentrationSum+=solverParams->nbhdConcLen*currentConcentration;
		scratch+=(1+solverParams->nbhdConcLen*currentDiffCoef)*currentConcentration;
	}

			
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	
	
	//concentrationSum*=currentDiffCoef;
	
	g_scratch[g_linearInd]=scratch;
	
}


/*

__kernel void vct_add(__global float* vector,
    __global const float * add,
	int size
    )
{
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int idx=get_global_id(0);

	if(idx>=size)
		return;

	vector[idx]+=add[idx];
}*/


float diffOp(float dx2, int nbhdLen, __constant int4 const * nbhdShifts, float currDiffCoeff, float currField,
	int4 lDim, int4 lInd, __local const float * l_field, bool bnd)
{
	float scratch=0.f;
	if(!bnd){
		for(int i=0; i<nbhdLen; ++i){
			int4 shift=nbhdShifts[i];
			int lShiftedInd=ext3DIndToLinear(lDim, lInd+shift);
			scratch+=l_field[lShiftedInd];
		}
		scratch-=nbhdLen*currField;
		
	}else
		scratch=currField; 

	return scratch*currDiffCoeff/dx2;
	
}

float diffOpDiag(float dx2, int nbhdLen, float currDiffCoeff, bool bnd){
	if(!bnd){
		float scratch=0.f;
		
		scratch-=nbhdLen; /* *currField;*/
		scratch*=currDiffCoeff;

		return scratch/dx2;
	}else
		return 1;
	
}

/*__kernel void newtonItKernel(float epsilon,
	__global const float* g_krylovVct,//1
	__global const float* g_newField,
	__global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__global float* g_scratch,//6
    __local float *l_field,
	__local unsigned char *l_cellType,
	float dt)
{

	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;

	int4 lInd={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int4 lDim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(lDim, lInd);//index of a current work item in a local space (in a block of shared memory)

	

	globToLocal(g_dim, g_ind, lDim, lInd, g_krylovVct, g_cellType, l_field, l_cellType);

	unsigned char currCelltype=l_cellType[l_linearInd];
	float currDiffCoef=solverParams->diffCoef[currCelltype];

	

	float argValue=l_field[l_linearInd];

	

	float newFieldVal=g_newField[g_linearInd];

	

	bool bnd=(g_ind.x==1||g_ind.x==solverParams->xDim||
		g_ind.y==1||g_ind.y==solverParams->yDim||
		g_ind.z==1||g_ind.z==solverParams->zDim);


	//g_scratch[g_linearInd]=diffOp(solverParams->dx2, solverParams->nbhdConcLen, nbhdConcShifts, currDiffCoef, argValue, lDim, lInd, l_field, bnd);
	//g_scratch[g_linearInd]=epsilon;
	
	//implicit scheme
	float dx2=solverParams->dx*solverParams->dx;
	g_scratch[g_linearInd]=argValue-dt*(diffOp(dx2, solverParams->nbhdConcLen, nbhdConcShifts, currDiffCoef, argValue, lDim, lInd, l_field, bnd)+
		(AdditionalTermFGF(newFieldVal+argValue*epsilon)-AdditionalTermFGF(newFieldVal))/epsilon);
	

}


__kernel void newtonItKernelDiag(float epsilon,
	__global const float* g_newField,
	__global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__global float* g_scratch, 
	float dt)
{
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	//int4 lInd={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	//int4 lDim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	
	//globToLocal(g_dim, g_ind, lDim, lInd, g_krylovVct, g_cellType, l_field, l_cellType);

	unsigned char currCelltype=g_cellType[g_linearInd];
	float currDiffCoef=solverParams->diffCoef[currCelltype];

	//float argValue=g_krylovVct[g_linearInd];

	float newFieldVal=g_newField[g_linearInd];

	bool bnd=(g_ind.x==1||g_ind.x==solverParams->xDim||
		g_ind.y==1||g_ind.y==solverParams->yDim||
		g_ind.z==1||g_ind.z==solverParams->zDim);
	
	float dx2=solverParams->dx*solverParams->dx;
	float res=1-dt*diffOpDiag(dx2, solverParams->nbhdConcLen, currDiffCoef, bnd);
	
	res+=dt*(AdditionalTermFGF(newFieldVal+epsilon)-AdditionalTermFGF(newFieldVal))/epsilon;

	if(fabs(res)<epsilon)
	{
		g_scratch[g_linearInd]=epsilon;
		return;
	}
	
	//implicit scheme
	g_scratch[g_linearInd]=1/res;
	
}

__kernel void fullExpr(int negate,
	__global const float* g_field,
	__global const float* g_argVector,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__global float* g_scratch,//6
    __local float *l_field,
	__local unsigned char *l_cellType, 
	float dt)
{
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	int4 lInd={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int4 lDim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	//g_scratch[10]=-nbhdConcShifts[0].x;
	//g_scratch[11]=nbhdConcShifts[0].x;
	//return;


	int l_linearInd=ext3DIndToLinear(lDim, lInd);//index of a current work item in a local space (in a block of shared memory)

	globToLocal(g_dim, g_ind, lDim, lInd, g_argVector, g_cellType, l_field, l_cellType);

	unsigned char currCelltype=l_cellType[l_linearInd];
	float currDiffCoef=solverParams->diffCoef[currCelltype];

	float argValue=l_field[l_linearInd];

	bool bnd=(g_ind.x==1||g_ind.x==solverParams->xDim||
		g_ind.y==1||g_ind.y==solverParams->yDim||
		g_ind.z==1||g_ind.z==solverParams->zDim);
	
	//implicit scheme
	float dx2=solverParams->dx*solverParams->dx;
	float res=argValue-g_field[g_linearInd]-
		dt*(diffOp(dx2, solverParams->nbhdConcLen, nbhdConcShifts, currDiffCoef, argValue, lDim, lInd, l_field, bnd)+
						  AdditionalTermFGF(argValue));

	//float res=argValue;

	if(negate)
		g_scratch[g_linearInd]=-res;
	else
		g_scratch[g_linearInd]=res;

}


//if hasDelta==true compute F(g_field+e*g_fieldDelta)
//otherwise compute F(g_fieldDelta)
//if negate==true negate the reult
__kernel void newtonKernel(int negate,
	__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType)//TODO: unify with a regular solver?
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)

	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];
	float concentrationSum=0.f;
	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];

	float scratch=0.f;
	
	for(int i=0; i<solverParams->nbhdConcLen; ++i){
		int4 shift=nbhdConcShifts[i];
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		scratch=l_field[lInd];
		//concentrationSum-=l_field[lInd];
	}
	scratch*=currentDiffCoef;
	
	scratch+=(1-solverParams->nbhdConcLen*currentDiffCoef)*currentConcentration;

	if(negate)
		g_scratch[g_linearInd]=-scratch;
	else
		g_scratch[g_linearInd]=scratch;
	
}

//if hasDelta==true compute F(g_field+e*g_fieldDelta)
//otherwise compute F(g_fieldDelta)
//if negate==true negate the reult
__kernel void newtonReactionKernel(float epsilon,
	__global const float* g_field,
    __global const float* g_fieldDelta,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
    __global float* g_scratch
    )
{
	
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space

	float currentConcentration=g_field[g_linearInd];
	float currentConcentrationDelta=g_fieldDelta[g_linearInd];
	
	float scratch=AdditionalTermFGF(currentConcentration+epsilon*currentConcentrationDelta);
	scratch-=AdditionalTermFGF(currentConcentration);
	
	g_scratch[g_linearInd]+=scratch/epsilon;
	
}*/


//for testing purposes
/*__kernel void diffHexagonal3D(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)


	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];

	float concentrationSum=0.f;


	for(int i=0; i<12; ++i){
		int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, 12);
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		concentrationSum+=l_field[lInd];
	}
	concentrationSum-=12*currentConcentration;
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;

	for(int i=0; i<6; ++i){
		int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, 6);
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
	}
	
	float dt_dx2=solverParams->dt/(solverParams->dx2);

	//float scratch=varDiffSumTerm;
    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-solverParams->dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	g_scratch[g_linearInd]=scratch;
	
}

//for testing purposes
__kernel void diffHexagonal2D1(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)


	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];

	float concentrationSum=0.f;

	for(int i=0; i<6; ++i){
		int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, 6);

	
		
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		concentrationSum+=l_field[lInd];
	}
	concentrationSum-=6*currentConcentration;
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;

	for(int i=0; i<3; ++i){
		int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, 3);
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
	}
	
	float dt_dx2=solverParams->dt/(solverParams->dx2);


	//float scratch=varDiffSumTerm;
    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-solverParams->dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	
	g_scratch[g_linearInd]=scratch;
	
}
//for testing purposes
int4 getShift2D(int4 pt, int ind,  __constant const int4 *nbhdConcEff, int offsetLen)
{
	int row=pt.y%2;
		
	return nbhdConcEff[row*offsetLen+ind];
}

//for testing purposes
__kernel void diffHexagonal2D(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)


	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];

	float concentrationSum=0.f;

	int yShifts[6]={0,1,1,0,-1,-1};
	int xShifts[6];
	if((g_ind.x+1)%2==0){
		xShifts[0]=-1; xShifts[1]=-1; xShifts[2]=0; xShifts[3]=1; xShifts[4]=0; xShifts[5]=-1;
	}else{
		xShifts[0]=-1; xShifts[1]=0; xShifts[2]=1; xShifts[3]=1; xShifts[4]=1; xShifts[5]=0;
	};


	for(int i=0; i<6; ++i){
		//int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, 6);

		int4 shift=g_ind+(int4)(xShifts[i], yShifts[i], 0, 0);

		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		concentrationSum+=l_field[lInd];
	}
	concentrationSum-=6*currentConcentration;
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;

	for(int i=0; i<3; ++i){
		//int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, 6);
		int4 shift=g_ind+(int4)(xShifts[i], yShifts[i], 0, 0);
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
	}
	
	float dt_dx2=solverParams->dt/(solverParams->dx2);

	//float scratch=varDiffSumTerm;
    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-solverParams->dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	g_scratch[g_linearInd]=scratch;
	
}
//for testing purposes
__kernel void diffCartesian3D(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)


	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];

	float concentrationSum=0.f;


	for(int i=0; i<6; ++i){
		int4 shift=nbhdConcShifts[i];
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		concentrationSum+=l_field[lInd];
	}
	concentrationSum-=6*currentConcentration;
	
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;
	for(int i=0; i<3; ++i){
		int lInd=ext3DIndToLinear(l_dim, l_ind+nbhdDiffShifts[i]);
		varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
	}
	
	
	float dt_dx2=solverParams->dt/(solverParams->dx2);

	//float scratch=varDiffSumTerm;
    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-solverParams->dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	g_scratch[g_linearInd]=scratch;
	
}
//for testing purposes
__kernel void diffCartesian2D(__global const float* g_field,
    __global const unsigned char * g_cellType,
	__global UniSolverParams_t  const *solverParams,
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts, 
    __global float* g_scratch,
    __local float *l_field,
	__local unsigned char *l_cellType
    )
{

	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
	int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};

	int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
	int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};

	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)


	globToLocal(g_dim, g_ind, l_dim, l_ind, g_field, g_cellType, l_field, l_cellType);

	float currentConcentration=l_field[l_linearInd];

	float concentrationSum=0.f;


	for(int i=0; i<4; ++i){
		int4 shift=nbhdConcShifts[i];
		int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
		concentrationSum+=l_field[lInd];
	}
	concentrationSum-=4*currentConcentration;
	
		
	//g_scratch[g_linearInd]=concentrationSum;
	//return;

	unsigned char curentCelltype=l_cellType[l_linearInd];
	float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	concentrationSum*=currentDiffCoef;
	
	float varDiffSumTerm=0.f;
	for(int i=0; i<2; ++i){
		int lInd=ext3DIndToLinear(l_dim, l_ind+nbhdDiffShifts[i]);
		varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
	}
	
	
	float dt_dx2=solverParams->dt/(solverParams->dx2);

	//float scratch=varDiffSumTerm;
    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-solverParams->dt*solverParams->decayCoef[curentCelltype])*currentConcentration;

	g_scratch[g_linearInd]=scratch;
	
}

//for testing purposes
inline
size_t glob2DIndToLinear(int dimX, int x, int y)
{
	return y*(dimX+2)+x;
}
//for testing purposes
inline
size_t block2DIndToLinear(int blockDimX,  
	int x, int y)
{
	return y*blockDimX+x;
}
//for testing purposes
__kernel void diff2DHex(__global const float* field_lr,
    __global const unsigned char * celltype_lr,
    __global float* scratch_lr,
    __global SolverParams_t  const *solverParams,
	__local float *fieldBlock,
    __local unsigned char *celltypeBlock,
    __local float *scratchBlock)
{
	size_t shift=(solverParams->dimx+2)*(solverParams->dimy+2);

	__global const float* field=field_lr+shift;
    __global const unsigned char * celltype=celltype_lr+shift;
    __global float* scratch=scratch_lr+shift;

    int bx=get_group_id(0);
    int by=get_group_id(1);
	
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    		
    int DIMX=solverParams->dimx;
    int DIMY=solverParams->dimy;
    	
	int const blockSizeX=get_local_size(0);
	int const blockSizeY=get_local_size(1);
		
	int x= bx*blockSizeX+tx;
    int y= by*blockSizeY+ty;
	

    //solve actual diff equation
    
    float dt_dx2=solverParams->dt/(solverParams->dx2);

    int curentCelltype=celltype[glob2DIndToLinear(DIMX, x+1, y+1)];

	int yShifts[6]={0,1,1,0,-1,-1};
	int xShifts[6];
	if(x%2==0){
		xShifts[0]=-1; xShifts[1]=-1; xShifts[2]=0; xShifts[3]=1; xShifts[4]=0; xShifts[5]=-1;
	}else{
		xShifts[0]=-1; xShifts[1]=0; xShifts[2]=1; xShifts[3]=1; xShifts[4]=1; xShifts[5]=0;
	};

	int nbhdInd[6];

	int gCurrInd=glob2DIndToLinear(DIMX, x+1, y+1);
	
	float concentrationSum =0.0f;
	for(int i=0; i<6; ++i){
		int xNbhdInd=x+1+xShifts[i];
		int yNbhdInd=y+1+yShifts[i];
		nbhdInd[i]=glob2DIndToLinear(DIMX, xNbhdInd, yNbhdInd);
		concentrationSum+=field[nbhdInd[i]];
	}
	concentrationSum-=6*field[gCurrInd];

	__global float const * diffCoef=solverParams->diffCoef;
    __global float const * decayCoef=solverParams->decayCoef;

	concentrationSum*=diffCoef[curentCelltype];
		
    float varDiffSumTerm=0.0f;

	for(int i=1; i<=3; ++i){
		int ni=nbhdInd[i];
		varDiffSumTerm+=(diffCoef[celltype[ni]]-diffCoef[curentCelltype])*(field[ni]-field[gCurrInd]);
	}

	scratch[gCurrInd]=dt_dx2*(concentrationSum+varDiffSumTerm)+(1-solverParams->dt*decayCoef[curentCelltype])*field[gCurrInd];

 
	 
}*/


