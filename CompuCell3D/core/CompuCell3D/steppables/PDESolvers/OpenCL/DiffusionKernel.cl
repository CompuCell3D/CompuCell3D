//#include "lib/CompuCell3DSteppables/OpenCL/GPUSolverParams.h"

//Ivan Komarov, Maciek Swat

//old version with int3 type used produced unexpected behavior, so the type was changed to int4 (April 2012)

// #pragma OPENCL EXTENSION cl_intel_printf : enable

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

__kernel void secreteSingleFieldKernel( __global float* g_field, __global const unsigned char * g_cellType, __global UniSolverParams_t  const *solverParams) {

    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space


    unsigned char cellType=g_cellType[g_linearInd];

    float secrConst=solverParams->secretionData[cellType][SECRETION_CONST];
    float currConc=g_field[g_linearInd];
    
    if (secrConst){
        g_field[g_linearInd]+=secrConst;
        // g_field[g_linearInd]=g_field[g_linearInd]+secrConst;
    }
    
    if (solverParams->secretionDoUptake){        
        float relativeUptake  = solverParams->secretionData[cellType][RELATIVE_UPTAKE];
        if (relativeUptake){
            float maxUptake=solverParams->secretionData[cellType][MAX_UPTAKE];            
            float relativeUptakeQuantity=currConc*relativeUptake;
            if (relativeUptakeQuantity>maxUptake){
                g_field[g_linearInd]-=maxUptake;
            
            }else{
                g_field[g_linearInd]-=relativeUptakeQuantity;
            }           
        }    
    }    
}


__kernel void secreteConstantConcentrationSingleFieldKernel( __global float* g_field, __global const unsigned char * g_cellType, __global UniSolverParams_t  const *solverParams) {

    
    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	

	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space


    unsigned char cellType=g_cellType[g_linearInd];

    float secrConst=solverParams->secretionConstantConcentrationData[cellType];
        
    if (secrConst){
        g_field[g_linearInd]=secrConst;

    }
    

}


__kernel void secreteOnContactSingleFieldKernel	( __global float* g_field, __global const unsigned char * g_cellType, __global  float * g_cellId,  __global UniSolverParams_t  const *solverParams,  __constant int4 const *nbhdConcShifts,  __local unsigned char *l_cellType, __local float *l_cellId) {

//interestingly this kernel when working on global device memory i.e. no copying anything into fast registers is faster than the kernel that uses local memory - I leave the other one commented out


       
    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;

	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
    unsigned char currentCellType = g_cellType[g_linearInd];    
    
    if (! solverParams->secretionOnContactData[currentCellType][UCHAR_MAX+1]){
        //all currentCellType secrete on contact secretion constants are 0
        return;
    }
    
    float currentCellId=g_cellId[g_linearInd];
    
    float currentConcentration=g_field[g_linearInd];
    float NON_CELL=-2.0; //we assume medium cell id is -1 not zero because normally cells in older versions of CC3D we allwoed cells with id 0 . For that reason we set NON_CEll to -2.0
    
	if(solverParams->hexLattice){
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
			int gInd=ext3DIndToLinear(g_dim, g_ind+shift);
            
            float n_cellId = g_cellId[gInd];
            
            if (n_cellId!=NON_CELL && currentCellId!=n_cellId) {
                unsigned char n_cellType = g_cellType[gInd];
                float secrConst=solverParams->secretionOnContactData[currentCellType][n_cellType];
                if (secrConst){
                    g_field[g_linearInd]=currentConcentration+secrConst;
                }            
            }            			
		}
		
	}else{
    
        
        
		for(int i=0; i<solverParams->nbhdConcLen; ++i){
			int4 shift=nbhdConcShifts[i];
			int gInd = ext3DIndToLinear(g_dim, g_ind+shift);
            
			float n_cellId = g_cellId[gInd];
            
            
            if (n_cellId!=NON_CELL && currentCellId!=n_cellId) {
                unsigned char n_cellType = g_cellType[gInd];
                float secrConst=solverParams->secretionOnContactData[currentCellType][n_cellType];
                if (secrConst){
                    g_field[g_linearInd]=currentConcentration+secrConst;
                    
                }
            
            }
            
            
		}    

	}    

}





__kernel void boundaryConditionInitLatticeCornersKernel( __global float* g_field, __global UniSolverParams_t  const *solverParams, __global BCSpecifier  const *bcSpecifier){
    
    int bx=get_group_id(0);
    int by=get_group_id(1);
    int bz=get_group_id(2);
    
    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    int tx=get_local_id(0);
    int ty=get_local_id(1);
    int tz=get_local_id(2);
    
    int l_dim_x=get_local_size(0);
    int l_dim_y=get_local_size(1);
    int l_dim_z=get_local_size(2);
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	// int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
    int4 l_ind_orig={tx,  ty, tz, 0};
    
    int4 l_ind={tx+1,  ty+1,  tz+1, 0};

	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
    
	// int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};
    int4 l_dim={l_dim_x,  l_dim_x,  l_dim_x, 0};

    //because enums are not supported in opencl kernels we simply instantiate them as regular variables
	//enum BCType
    int PERIODIC=0,CONSTANT_VALUE=1,CONSTANT_DERIVATIVE=2;
    int MIN_X=0,MAX_X=1,MIN_Y=2,MAX_Y=3,MIN_Z=4,MAX_Z=5;
    
    
        // X axis
    float deltaX=solverParams->dx;    


    
    //X-axis
    if (bcSpecifier->planePositions[MIN_X] == PERIODIC|| bcSpecifier->planePositions[MAX_X]==PERIODIC){

        if (gid_x==0 && gid_y==0 && gid_z==0){
            for (int z = 0 ; z < g_dim.z+2 ; ++z){
                g_field[ext3DIndToLinear(g_dim, (int4)(0,gid_y,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x,gid_y,z,0))];
            }
        }

        if (gid_x==0 && gid_y==g_dim.y-1 && gid_z==0){
            for (int z = 0 ; z < g_dim.z+2 ; ++z){
                g_field[ext3DIndToLinear(g_dim, (int4)(0,gid_y+2,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x,gid_y+2,z,0))];
            }
        }

        if (gid_x==g_dim.x-1 && gid_y==0 && gid_z==0){
            for (int z = 0 ; z < g_dim.z+2 ; ++z){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y,z,0))];
            }
        }

        if (gid_x==g_dim.x-1 && gid_y==g_dim.y-1 && gid_z==0){
            for (int z = 0 ; z < g_dim.z+2 ; ++z){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+2,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y+2,z,0))];
            }
        }       
    }else{ // this part of bc init for hex lattice does not fully make sense but is equivalent to CPU code thus for now , to have two codes behave in similar fashion I keep it like that
        if (gid_x==0 && gid_y==0 && gid_z==0){
            if (bcSpecifier->planePositions[MIN_X]==CONSTANT_VALUE){    
            
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y,z,0))]=bcSpecifier->values[MIN_X];
                }
            }else if (bcSpecifier->planePositions[MIN_X]==CONSTANT_DERIVATIVE){ 
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y,z,0))]-bcSpecifier->values[MIN_X]*deltaX;                    
                }            
            }
        }
      
    
        if (gid_x==0 && gid_y==g_dim.y-1 && gid_z==0){
            if (bcSpecifier->planePositions[MIN_X]==CONSTANT_VALUE){    
            
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+2,z,0))]=bcSpecifier->values[MIN_X];
                }
            }else if (bcSpecifier->planePositions[MIN_X]==CONSTANT_DERIVATIVE){ 
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+2,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y+2,z,0))]-bcSpecifier->values[MIN_X]*deltaX;                    
                }            
            }
        }
      

        if (gid_x==g_dim.x-1 && gid_y==0 && gid_z==0){
            if (bcSpecifier->planePositions[MAX_X]==CONSTANT_VALUE){    
            
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y,z,0))]=bcSpecifier->values[MAX_X];
                }
            }else if (bcSpecifier->planePositions[MAX_X]==CONSTANT_DERIVATIVE){ 
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,z,0))]+bcSpecifier->values[MAX_X]*deltaX;                    
                }            
            }
        }      
      
        if (gid_x==g_dim.x-1 && gid_y==g_dim.y-1 && gid_z==0){
            if (bcSpecifier->planePositions[MAX_X]==CONSTANT_VALUE){    
            
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+2,z,0))]=bcSpecifier->values[MAX_X];
                }
            }else if (bcSpecifier->planePositions[MAX_X]==CONSTANT_DERIVATIVE){ 
                for (int z = 0 ; z < g_dim.z+2 ; ++z){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+2,z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,z,0))]+bcSpecifier->values[MAX_X]*deltaX;                    
                }            
            }
        }   
      
    }
     
    //Y-axis
    if (bcSpecifier->planePositions[MIN_Y] == PERIODIC|| bcSpecifier->planePositions[MAX_Y]==PERIODIC){
        if (gid_y==0 && gid_z==0 && gid_x==0){
            for (int x = 0 ; x < g_dim.x+2 ; ++x){
                g_field[ext3DIndToLinear(g_dim, (int4)(x,0,gid_z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(x,g_dim.y,gid_z,0))];
            }
        }
        
        if (gid_y==0 && gid_z==g_dim.z-1 && gid_x==0){
            for (int x = 0 ; x < g_dim.x+2 ; ++x){
                g_field[ext3DIndToLinear(g_dim, (int4)(x,0,gid_z+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(x,g_dim.y,gid_z+2,0))];
            }
        }

        if (gid_y==g_dim.y-1 && gid_z==0 && gid_x==0){
            for (int x = 0 ; x < g_dim.x+2 ; ++x){
                g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(x,1,gid_z,0))];
            }
        }

        if (gid_y==g_dim.y-1 && gid_z==g_dim.z-1 && gid_x==0){
            for (int x = 0 ; x < g_dim.x+2 ; ++x){
                g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(x,1,gid_z+2,0))];
            }
        }
        
    }else{
    
        if (gid_y==0 && gid_z==0 && gid_x==0){
            if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_VALUE){    
            
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y,gid_z,0))] = bcSpecifier->values[MIN_Y];
                }
            }else if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_DERIVATIVE){ 
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y,gid_z,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(x,1,gid_z,0))]-bcSpecifier->values[MIN_Y]*deltaX;                    
                }            
            }
        }    

        if (gid_y==0 && gid_z==g_dim.z-1 && gid_x==0){
            if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_VALUE){    
            
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y,gid_z+2,0))] = bcSpecifier->values[MIN_Y];
                }
            }else if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_DERIVATIVE){ 
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y,gid_z+2,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(x,1,gid_z+2,0))]-bcSpecifier->values[MIN_Y]*deltaX;                    
                }            
            }
        }

        if (gid_y==g_dim.y-1 && gid_z==0 && gid_x==0){
            if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_VALUE){    
            
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z,0))] = bcSpecifier->values[MAX_Y];
                }
            }else if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_DERIVATIVE){ 
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+1,gid_z,0))]+bcSpecifier->values[MAX_Y]*deltaX;                    
                }            
            }
        }

        if (gid_y==g_dim.y-1 && gid_z==g_dim.z-1 && gid_x==0){
            if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_VALUE){    
            
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z+2,0))] = bcSpecifier->values[MAX_Y];
                }
            }else if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_DERIVATIVE){ 
                for (int x = 0 ; x < g_dim.x+2 ; ++x){
                    g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+2,gid_z+2,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(x,gid_y+1,gid_z+2,0))]+bcSpecifier->values[MAX_Y]*deltaX;                    
                }            
            }
        }
        
    }

    //Z-axis    
    if (bcSpecifier->planePositions[MIN_Z] == PERIODIC || bcSpecifier->planePositions[MAX_Z]==PERIODIC){
        if (gid_z==0 && gid_x==0 && gid_y==0){
            for (int y = 0 ; y < g_dim.y+2 ; ++y){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,0,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,g_dim.z,0))];
            }
        }
        
        if (gid_z==0 && gid_x==g_dim.x-1 && gid_y==0){
            for (int y = 0 ; y < g_dim.y+2 ; ++y){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,0,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,g_dim.z,0))];
            }
        }

        if ( gid_z==g_dim.z-1 && gid_x==0 && gid_y==0){
            for (int y = 0 ; y < g_dim.y+2 ; ++y){
                g_field[ext3DIndToLinear(g_dim, (int4)(0,y,gid_z+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(0,y,1,0))];
            }
        }

        if ( gid_z==g_dim.z-1 && gid_x==g_dim.x-1 && gid_y==0){
            for (int y = 0 ; y < g_dim.y+2 ; ++y){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,1,0))];
            }
        }
        
    }else{
    
        if (gid_z==0 && gid_x==0 && gid_y==0){
            if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_VALUE){    
            
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,gid_z,0))] = bcSpecifier->values[MIN_Z];
                }
            }else if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_DERIVATIVE){ 
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,gid_z,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,1,0))]-bcSpecifier->values[MIN_Z]*deltaX;                    
                }            
            }
        }     

        if (gid_z==0 && gid_x==g_dim.x-1 && gid_y==0){
            if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_VALUE){    
            
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z,0))] = bcSpecifier->values[MIN_Z];
                }
            }else if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_DERIVATIVE){ 
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,1,0))]-bcSpecifier->values[MIN_Z]*deltaX;                    
                }            
            }
        } 
                
        if ( gid_z==g_dim.z-1 && gid_x==0 && gid_y==0){
            if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_VALUE){                
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,gid_z+2,0))] = bcSpecifier->values[MAX_Z];
                }
            }else if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_DERIVATIVE){ 
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,gid_z+2,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,y,gid_z+1,0))]+bcSpecifier->values[MAX_Z]*deltaX;                    
                }            
            }
        } 
                
        if ( gid_z==g_dim.z-1 && gid_x==g_dim.x-1 && gid_y==0){
            if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_VALUE){                
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z+2,0))] = bcSpecifier->values[MAX_Z];
                }
            }else if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_DERIVATIVE){ 
                for (int y = 0 ; y < g_dim.y+2 ; ++y){
                    g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z+2,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,y,gid_z+1,0))]+bcSpecifier->values[MAX_Z]*deltaX;                    
                }            
            }
        } 
        
        
    }
    
    

}              


__kernel void myKernel(__global float* g_field, __global UniSolverParams_t  const *solverParams, __global BCSpecifier  const *bcSpecifier) 
{                
    int bx=get_group_id(0);
    int by=get_group_id(1);
    int bz=get_group_id(2);
}



__kernel void boundaryConditionInitKernel( __global float* g_field, __global UniSolverParams_t  const *solverParams, __global BCSpecifier  const *bcSpecifier)           

{

    
    
    int bx=get_group_id(0);
    int by=get_group_id(1);
    int bz=get_group_id(2);
    
    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    int tx=get_local_id(0);
    int ty=get_local_id(1);
    int tz=get_local_id(2);
    
    int l_dim_x=get_local_size(0);
    int l_dim_y=get_local_size(1);
    int l_dim_z=get_local_size(2);
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions
	// int4 g_ind={get_global_id(0)+1, get_global_id(1)+1, get_global_id(2)+1, 0};
    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};
    
	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
    int4 l_ind_orig={tx,  ty, tz, 0};
    
	// int4 l_ind={get_local_id(0)+1,  get_local_id(1)+1,  get_local_id(2)+1, 0};
    int4 l_ind={tx+1,  ty+1,  tz+1, 0};

	// int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
    
	// int4 l_dim={get_local_size(0),  get_local_size(1),  get_local_size(2), 0};
    int4 l_dim={l_dim_x,  l_dim_x,  l_dim_x, 0};

    //because enums are not supported in opencl kernels we simply instantiate them as regular variables
	//enum BCType
    int PERIODIC=0,CONSTANT_VALUE=1,CONSTANT_DERIVATIVE=2;
    int MIN_X=0,MAX_X=1,MIN_Y=2,MAX_Y=3,MIN_Z=4,MAX_Z=5;
    

    
    //boundaryConditionInit
        // X axis
    float deltaX=solverParams->dx;    
    
    // g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0))]=bcSpecifier->values[MIN_X];
    // return;
    // bcSpecifier->planePositions[MAX_X];
    // return;
    //x- axis
    if (bcSpecifier->planePositions[MIN_X] == PERIODIC|| bcSpecifier->planePositions[MAX_X]==PERIODIC){
        
        // return;
        if (gid_x==0){
        //periodic bc x        
            g_field[ext3DIndToLinear(g_dim, (int4)(0,gid_y+1,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x,gid_y+1,gid_z+1,0))];
            
        }
        
      if (gid_x==g_dim.x-1){
        //periodic bc x
            g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x+1,gid_y+1,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y+1,gid_z+1,0))];            
        }

        
    }else{
        if (gid_x==0){
            if (bcSpecifier->planePositions[MIN_X]==CONSTANT_VALUE){    
                g_field[ext3DIndToLinear(g_dim, (int4)(0,gid_y+1,gid_z+1,0))]=bcSpecifier->values[MIN_X];
                
            }else if (bcSpecifier->planePositions[MIN_X]==CONSTANT_DERIVATIVE){ 
                g_field[ext3DIndToLinear(g_dim, (int4)(0,gid_y+1,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(1,gid_y+1,gid_z+1,0))]-bcSpecifier->values[MIN_X]*deltaX;
            }
        }
        
        else if (gid_x==g_dim.x-1){ 
            if (bcSpecifier->planePositions[MAX_X]==CONSTANT_VALUE){    
                g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x+1,gid_y+1,gid_z+1,0))]=bcSpecifier->values[MAX_X];
                
            }else if (bcSpecifier->planePositions[MAX_X]==CONSTANT_DERIVATIVE){ 
                g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x+1 , gid_y+1,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(g_dim.x , gid_y+1,gid_z+1,0))]+bcSpecifier->values[MAX_X]*deltaX;
            }
        
        }
    }
    
     barrier(CLK_LOCAL_MEM_FENCE); 
    //y- axis
    if (bcSpecifier->planePositions[MIN_Y] == PERIODIC|| bcSpecifier->planePositions[MAX_Y]==PERIODIC){    
        if (gid_y==0){
        //periodic bc y
            g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,0,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,g_dim.y , gid_z+1,0))];
        }
        
        if (gid_y==g_dim.y-1){
        //periodic bc y
            g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,g_dim.y+1,gid_z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,1,gid_z+1,0))];
            
        } 
    
    }else{
        if (gid_y==0){
            if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_VALUE){  
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,0,gid_z+1,0))]=bcSpecifier->values[MIN_Y];            
            }else if (bcSpecifier->planePositions[MIN_Y]==CONSTANT_DERIVATIVE){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,0,gid_z+1,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,1,gid_z+1,0))]-bcSpecifier->values[MIN_Y]*deltaX;                          
            }
        }
        
        else if (gid_y==g_dim.y-1){ 
            if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_VALUE){ 
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,g_dim.y+1,gid_z+1,0))]=bcSpecifier->values[MAX_Y];                                            
            }else if (bcSpecifier->planePositions[MAX_Y]==CONSTANT_DERIVATIVE){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,g_dim.y+1,gid_z+1,0))] = g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,g_dim.y  ,gid_z+1,0))]+bcSpecifier->values[MAX_Y]*deltaX;                           
            }        
        }    
    }
    
     barrier(CLK_LOCAL_MEM_FENCE); 
        //z- axis
    if (bcSpecifier->planePositions[MIN_Z] == PERIODIC|| bcSpecifier->planePositions[MAX_Z]==PERIODIC){    
        if (gid_z==0){
        //periodic bc z
            g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,0,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,g_dim.z,0))];
        }
        
        if (gid_z==g_dim.z-1){
        //periodic bc z
            g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,g_dim.z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,1,0))];        
        }   
    
    }else{
        if (gid_z==0){
            if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_VALUE){  
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,0,0))]=bcSpecifier->values[MIN_Z];            
            }else if (bcSpecifier->planePositions[MIN_Z]==CONSTANT_DERIVATIVE){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,0,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,1,0))]-bcSpecifier->values[MIN_Z]*deltaX;                
            }
        }
        
        else if (gid_z==g_dim.z-1){ 
            if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_VALUE){             
                 g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,g_dim.z+1,0))]=bcSpecifier->values[MAX_Z];                                            
            }else if (bcSpecifier->planePositions[MAX_Z]==CONSTANT_DERIVATIVE){
                g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,g_dim.z+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,g_dim.z,0))]+bcSpecifier->values[MAX_Z]*deltaX;
            }        
        }    
    }
    
    
 
    
}    


__kernel void uniDiff(__global float* g_field,
    __global float* g_scratch,    
	__global UniSolverParams_t  const *solverParams,
    __global BCSpecifier  const *bcSpecifier,
    __global const unsigned char * g_cellType,    
    __global const signed char *g_bcIndicator,        
	__constant int4 const *nbhdConcShifts, 
	__constant int4 const *nbhdDiffShifts,     
    __local float *l_field,
	__local unsigned char *l_cellType
    )    
{
    //because enums are not supported in opencl kernels we simply instantiate them as regular variables
	//enum BCType
    int PERIODIC=0,CONSTANT_VALUE=1,CONSTANT_DERIVATIVE=2;    
    int INTERNAL=-2,BOUNDARY=-1,MIN_X=0,MAX_X=1,MIN_Y=2,MAX_Y=3,MIN_Z=4,MAX_Z=5;
    
    __local float bcMultFactor[2];
    
    bcMultFactor[0]=-1;
    bcMultFactor[1]=+1;
    
    int bx=get_group_id(0);
    int by=get_group_id(1);
    int bz=get_group_id(2);
    
    int gid_x=get_global_id(0);
    int gid_y=get_global_id(1);
    int gid_z=get_global_id(2);
    
    int tx=get_local_id(0);
    int ty=get_local_id(1);
    int tz=get_local_id(2);
    
    int l_dim_x=get_local_size(0);
    int l_dim_y=get_local_size(1);
    int l_dim_z=get_local_size(2);
    
    float deltaX=solverParams->dx;
    
	//there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions

    int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		return;
	
    int4 l_ind_orig={tx,  ty, tz, 0};
    

    int4 l_ind={tx+1,  ty+1,  tz+1, 0};

	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
    

    int4 l_dim={l_dim_x,  l_dim_x,  l_dim_x, 0};

        
	int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)
    
    // moving data from global to local
    l_field[l_linearInd]=g_field[g_linearInd];
    l_cellType[l_linearInd]=g_cellType[g_linearInd];
        
    barrier(CLK_LOCAL_MEM_FENCE);     
    
    
    if(solverParams->hexLattice){
        //this part is slow - essentially we do unnecessary multiple copies of data but the code is simple to uinderstand. I tried doing series of if-else statements in addition to copying faces but this always resulted in runtime-error. 
        // I leave it in it for now, maybe later I can debug it better
        int4 corner_offset1[26]={(int4)(1,0,0,0) , (int4)(1,1,0,0) , (int4)(0,1,0,0) , (int4)(-1,1,0,0) , (int4)(-1,0,0,0) , (int4)(-1,-1,0,0) , (int4)(0,-1,0,0) , (int4)(1,-1,0,0),
        (int4)(1,0,1,0) , (int4)(1,1,1,0) , (int4)(0,1,1,0) , (int4)(-1,1,1,0) , (int4)(-1,0,1,0) , (int4)(-1,-1,1,0) , (int4)(0,-1,1,0) , (int4)(1,-1,1,0) , (int4)(0,0,1,0),
        (int4)(1,0,-1,0) , (int4)(1,1,-1,0) , (int4)(0,1,-1,0) , (int4)(-1,1,-1,0) , (int4)(-1,0,-1,0) , (int4)(-1,-1,-1,0) , (int4)(0,-1,-1,0) , (int4)(1,-1,-1,0) , (int4)(0,0,-1,0)         
        };     
        
        for (int i = 0 ; i < 26 ; ++i){
       
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            // barrier(CLK_LOCAL_MEM_FENCE);     
        }    
        
        
    }else{


        
        if (tx==0){
            l_field[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
        }
        
        // have to use extra condition gid_x==solverParams->xDim-1 because then xDimension of the lattice is not a multiple f l_dim than 
        //tx==l_dim_x-1 will never trigger! This would cause problems with handling boundary conditions
        
        if (tx==l_dim_x-1 || gid_x==solverParams->xDim-1){ 
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
        }
        
        // if (gid_x==solverParams->xDim-1){
            // l_field[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];;        
            // // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0))];        
            // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0))];                    
        // }
      
        if (ty==0){
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
        }
        
      
        if (ty==l_dim_y-1 || gid_y==solverParams->yDim-1){
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
        }
      
        if (tz==0){
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
        }
      
        if (tz==l_dim_z-1 || gid_z==solverParams->zDim-1){
            l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
            l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
        }
    }    
    
    barrier(CLK_LOCAL_MEM_FENCE);     

    
	float currentConcentration=l_field[l_linearInd];
	float concentrationSum=0.f;
    float c_offset=0.f;
        
    
    //var Diffusion coef part
    unsigned char curentCelltype=l_cellType[l_linearInd];
    float currentDiffCoef=solverParams->diffCoef[curentCelltype];
    float varDiffSumTerm=0.f;    
    
    if (g_bcIndicator[g_linearInd]==INTERNAL){
    

        for(int i=0; i<solverParams->nbhdConcLen; ++i){
            int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
            int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
            concentrationSum+=l_field[lShiftedInd];
        }
        concentrationSum-=solverParams->nbhdConcLen*currentConcentration;    
        concentrationSum*=currentDiffCoef;

        //var Diffusion coef part         
        for(int i=0; i<solverParams->nbhdConcLen; ++i){
            int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
            int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
            // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef)*(l_field[lShiftedInd]-currentConcentration);
            varDiffSumTerm+=solverParams->diffCoef[l_cellType[lShiftedInd]]*(l_field[lShiftedInd]-currentConcentration);
        }          
        concentrationSum/=2.0;
        varDiffSumTerm/=2.0;
        
    }
    else{
            
            for(int i=0; i<solverParams->nbhdConcLen; ++i){
                int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
                int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);                
                signed char nBcIndicator = g_bcIndicator[ext3DIndToLinear(g_dim, g_ind+shift)];                    
                
                if (nBcIndicator==INTERNAL || nBcIndicator==BOUNDARY){ 

                    concentrationSum+=l_field[lShiftedInd];     
                 }else{
                    
                    if (bcSpecifier->planePositions[nBcIndicator]==PERIODIC){
                        concentrationSum+=l_field[lShiftedInd];     
                        
                    }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_VALUE){
                        concentrationSum += bcSpecifier->values[nBcIndicator];
                    }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_DERIVATIVE){
                        // CPU CODE was somethign like :
                        // if (nBcIndicator==BoundaryConditionSpecifier::MIN_X || nBcIndicator==BoundaryConditionSpecifier::MIN_Y || nBcIndicator==BoundaryConditionSpecifier::MIN_Z){ // for "left hand side" edges of the lattice the sign of the derivative expression is '-'
                        
                        concentrationSum += l_field[l_linearInd] +  bcMultFactor[nBcIndicator%2]*bcSpecifier->values[nBcIndicator]*deltaX;

                    }                    
                }
                

                
            }
            concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
            concentrationSum*=currentDiffCoef;
            
            //var Diffusion coef part
            for(int i=0; i<solverParams->nbhdConcLen; ++i){
                int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
                int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
                signed char nBcIndicator = g_bcIndicator[ext3DIndToLinear(g_dim, g_ind+shift)];                    
                c_offset=l_field[lShiftedInd];
                
                 if (!(nBcIndicator==INTERNAL) && !(nBcIndicator==BOUNDARY)){ 
                 
                    //for periodic bc we keep c_offset value assigned above     

                    if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_VALUE){
                        c_offset=bcSpecifier->values[nBcIndicator];                                            
                    }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_DERIVATIVE){
                        c_offset= currentConcentration + bcMultFactor[nBcIndicator%2]*bcSpecifier->values[nBcIndicator];                        
                    }                    
                 }                    
                 varDiffSumTerm += solverParams->diffCoef[l_cellType[lShiftedInd]]*(c_offset-currentConcentration);                       
            }          

            concentrationSum/=2.0;
            varDiffSumTerm/=2.0;                

	}



    
	float dx2=solverParams->dx*solverParams->dx;
    float dt=solverParams->dx*solverParams->dt;
	float dt_dx2=dt/dx2;
    

    float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-dt*solverParams->decayCoef[curentCelltype])*currentConcentration;
    
    g_scratch[g_linearInd]=scratch;
    
            
}



// // // __kernel void uniDiff(__global float* g_field,
    // // // __global float* g_scratch,    
	// // // __global UniSolverParams_t  const *solverParams,
    // // // __global BCSpecifier  const *bcSpecifier,
    // // // __global const unsigned char * g_cellType,    
    // // // __global const signed char *g_bcIndicator,        
	// // // __constant int4 const *nbhdConcShifts, 
	// // // __constant int4 const *nbhdDiffShifts,     
    // // // __local float *l_field,
	// // // __local unsigned char *l_cellType
    // // // )    
// // // {
    // // // //because enums are not supported in opencl kernels we simply instantiate them as regular variables
	// // // //enum BCType
    // // // int PERIODIC=0,CONSTANT_VALUE=1,CONSTANT_DERIVATIVE=2;    
    // // // int INTERNAL=-2,BOUNDARY=-1,MIN_X=0,MAX_X=1,MIN_Y=2,MAX_Y=3,MIN_Z=4,MAX_Z=5;
    
    // // // __local float bcMultFactor[2];
    
    // // // bcMultFactor[0]=-1;
    // // // bcMultFactor[1]=+1;
    
    // // // int bx=get_group_id(0);
    // // // int by=get_group_id(1);
    // // // int bz=get_group_id(2);
    
    // // // int gid_x=get_global_id(0);
    // // // int gid_y=get_global_id(1);
    // // // int gid_z=get_global_id(2);
    
    // // // int tx=get_local_id(0);
    // // // int ty=get_local_id(1);
    // // // int tz=get_local_id(2);
    
    // // // int l_dim_x=get_local_size(0);
    // // // int l_dim_y=get_local_size(1);
    // // // int l_dim_z=get_local_size(2);
    
    // // // float deltaX=solverParams->dx;
    
	// // // //there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions

    // // // int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	// // // if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		// // // return;
	
    // // // int4 l_ind_orig={tx,  ty, tz, 0};
    

    // // // int4 l_ind={tx+1,  ty+1,  tz+1, 0};

	// // // // // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    // // // int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
    

    // // // int4 l_dim={l_dim_x,  l_dim_x,  l_dim_x, 0};

        
	// // // int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	// // // int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)
    
    // // // // moving data from global to local
    // // // l_field[l_linearInd]=g_field[g_linearInd];
    // // // l_cellType[l_linearInd]=g_cellType[g_linearInd];
        
    // // // barrier(CLK_LOCAL_MEM_FENCE);     
    
    
    // // // if(solverParams->hexLattice){
        // // // //this part is slow - essentially we do unnecessary multiple copies of data but the code is simple to uinderstand. I tried doing series of if-else statements in addition to copying faces but this always resulted in runtime-error. 
        // // // // I leave it in it for now, maybe later I can debug it better
        // // // int4 corner_offset1[26]={(int4)(1,0,0,0) , (int4)(1,1,0,0) , (int4)(0,1,0,0) , (int4)(-1,1,0,0) , (int4)(-1,0,0,0) , (int4)(-1,-1,0,0) , (int4)(0,-1,0,0) , (int4)(1,-1,0,0),
        // // // (int4)(1,0,1,0) , (int4)(1,1,1,0) , (int4)(0,1,1,0) , (int4)(-1,1,1,0) , (int4)(-1,0,1,0) , (int4)(-1,-1,1,0) , (int4)(0,-1,1,0) , (int4)(1,-1,1,0) , (int4)(0,0,1,0),
        // // // (int4)(1,0,-1,0) , (int4)(1,1,-1,0) , (int4)(0,1,-1,0) , (int4)(-1,1,-1,0) , (int4)(-1,0,-1,0) , (int4)(-1,-1,-1,0) , (int4)(0,-1,-1,0) , (int4)(1,-1,-1,0) , (int4)(0,0,-1,0)         
        // // // };     
        
        // // // for (int i = 0 ; i < 26 ; ++i){
       
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            // // // // barrier(CLK_LOCAL_MEM_FENCE);     
        // // // }    
        
        
    // // // }else{
        // // // //copying faces
        // // // if (tx==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
        // // // }
        
        // // // if (tx==l_dim_x-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
        // // // }
        
      
        // // // if (ty==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
        // // // }
        
      
        // // // if (ty==l_dim_y-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
        // // // }
      
        // // // if (tz==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
        // // // }
      
        // // // if (tz==l_dim_z-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
        // // // }
    // // // }    
    
    // // // barrier(CLK_LOCAL_MEM_FENCE);     

    
	// // // float currentConcentration=l_field[l_linearInd];
	// // // float concentrationSum=0.f;
        
    
    // // // //var Diffusion coef part
    // // // unsigned char curentCelltype=l_cellType[l_linearInd];
    // // // float currentDiffCoef=solverParams->diffCoef[curentCelltype];
    // // // float varDiffSumTerm=0.f;    
    
    // // // if (g_bcIndicator[g_linearInd]==INTERNAL){
    // // // // if (true){

        // // // for(int i=0; i<solverParams->nbhdConcLen; ++i){
            // // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
            // // // int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
            // // // concentrationSum+=l_field[lShiftedInd];
        // // // }
        // // // concentrationSum-=solverParams->nbhdConcLen*currentConcentration;    
        // // // concentrationSum*=currentDiffCoef;

            // // // //var Diffusion coef part
        // // // for(int i=0; i<solverParams->nbhdDiffLen; ++i){
            // // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, solverParams->nbhdDiffLen);
            // // // int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
            // // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef)*(l_field[lShiftedInd]-currentConcentration);
        // // // }            
        
        
    // // // }
    // // // else{
            
            // // // for(int i=0; i<solverParams->nbhdConcLen; ++i){
                // // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
                // // // int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);                
                // // // signed char nBcIndicator = g_bcIndicator[ext3DIndToLinear(g_dim, g_ind+shift)];                    
                
                // // // if (nBcIndicator==INTERNAL || nBcIndicator==BOUNDARY){ 

                    // // // concentrationSum+=l_field[lShiftedInd];     
                 // // // }else{
                    
                    // // // if (bcSpecifier->planePositions[nBcIndicator]==PERIODIC){
                        // // // concentrationSum+=l_field[lShiftedInd];     
                        
                    // // // }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_VALUE){
                        // // // concentrationSum += bcSpecifier->values[nBcIndicator];
                    // // // }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_DERIVATIVE){
                        // // // // CPU CODE was somethign like :
                        // // // // if (nBcIndicator==BoundaryConditionSpecifier::MIN_X || nBcIndicator==BoundaryConditionSpecifier::MIN_Y || nBcIndicator==BoundaryConditionSpecifier::MIN_Z){ // for "left hand side" edges of the lattice the sign of the derivative expression is '-'
                        
                        // // // concentrationSum += l_field[l_linearInd] +  bcMultFactor[nBcIndicator%2]*bcSpecifier->values[nBcIndicator]*deltaX;

                    // // // }                    
                // // // }
                

                
            // // // }
            // // // concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
            // // // concentrationSum*=currentDiffCoef;
            
            // // // //var Diffusion coef part
            
            // // // for(int i=0; i<solverParams->nbhdDiffLen; ++i){
                // // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, solverParams->nbhdDiffLen);
                // // // int lShiftedInd=ext3DIndToLinear(l_dim, l_ind+shift);
                // // // signed char nBcIndicator = g_bcIndicator[ext3DIndToLinear(g_dim, g_ind+shift)];                    
                
                 // // // if (nBcIndicator==INTERNAL || nBcIndicator==BOUNDARY){ 
                    // // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef)*(l_field[lShiftedInd]-currentConcentration);  
                    // // // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef);  
                    // // // // varDiffSumTerm+=l_field[lShiftedInd];
                 // // // }else{
                    
                    // // // if (bcSpecifier->planePositions[nBcIndicator]==PERIODIC){
                        // // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef)*(l_field[lShiftedInd]-currentConcentration);     
                        
                    // // // }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_VALUE){
                    
                        // // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lShiftedInd]]-currentDiffCoef)*(bcSpecifier->values[nBcIndicator]-currentConcentration);
                    // // // }else if (bcSpecifier->planePositions[nBcIndicator]==CONSTANT_DERIVATIVE){
                        // // // //for non-periodic bc diff coeff of neighbor pixels which are in the external boundary os the same as diff coef of current pixel. hece no extra term here

                    // // // }                    
                // // // }                    
                
                
            // // // }
	// // // }



    
	// // // float dx2=solverParams->dx*solverParams->dx;
    // // // float dt=solverParams->dx*solverParams->dt;
	// // // float dt_dx2=dt/dx2;
    

    // // // float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-dt*solverParams->decayCoef[curentCelltype])*currentConcentration;
    
    // // // g_scratch[g_linearInd]=scratch;
    
            
// // // }





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

	// // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 

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



// old kernel - worked fine for cartesian lattice had issues on hex with non-operiodic bc

// // // // __kernel void uniDiff(__global float* g_field,
    // // // // __global const unsigned char * g_cellType,
	// // // // __global UniSolverParams_t  const *solverParams,
	// // // // __constant int4 const *nbhdConcShifts, 
	// // // // __constant int4 const *nbhdDiffShifts, 
    // // // // __global float* g_scratch,
    // // // // __local float *l_field,
	// // // // __local unsigned char *l_cellType,
	// // // // float dt
    // // // // )    
    
// // // __kernel void uniDiff(__global float* g_field,
    // // // __global float* g_scratch,
    // // // __global UniSolverParams_t  const *solverParams,
    // // // __global const unsigned char * g_cellType,	
	// // // __constant int4 const *nbhdConcShifts, 
	// // // __constant int4 const *nbhdDiffShifts,     
    // // // __local float *l_field,
	// // // __local unsigned char *l_cellType	
    // // // )      
// // // {
    
    // // // int bx=get_group_id(0);
    // // // int by=get_group_id(1);
    // // // int bz=get_group_id(2);
    
    // // // int gid_x=get_global_id(0);
    // // // int gid_y=get_global_id(1);
    // // // int gid_z=get_global_id(2);
    
    // // // int tx=get_local_id(0);
    // // // int ty=get_local_id(1);
    // // // int tz=get_local_id(2);
    
    // // // int l_dim_x=get_local_size(0);
    // // // int l_dim_y=get_local_size(1);
    // // // int l_dim_z=get_local_size(2);
    
	// // // //there is a border of 1 pixel around the domain, so shifting by 1 at all dimensions

    // // // int4 g_ind={gid_x+1, gid_y+1, gid_z+1, 0};

	// // // if(g_ind.x>solverParams->xDim||g_ind.y>solverParams->yDim||g_ind.z>solverParams->zDim)
		// // // return;
	
    // // // int4 l_ind_orig={tx,  ty, tz, 0};
    

    // // // int4 l_ind={tx+1,  ty+1,  tz+1, 0};

	// // // // // // int4 g_dim={get_global_size(0), get_global_size(1), get_global_size(2), 0};
    // // // int4 g_dim={solverParams->xDim, solverParams->yDim, solverParams->zDim, 0}; // g_dim has to have dimensions of field not of the workSize - after all we use g_dim to access arrays and those are indexed using user specified lattice dim , not work size 
    
    

    // // // int4 l_dim={l_dim_x,  l_dim_x,  l_dim_x, 0};

        
	// // // int g_linearInd=ext3DIndToLinear(g_dim, g_ind);//index of a current work item in a global space
	// // // int l_linearInd=ext3DIndToLinear(l_dim, l_ind);//index of a current work item in a local space (in a block of shared memory)
    
    // // // // moving data from global to local
    // // // l_field[l_linearInd]=g_field[g_linearInd];
    // // // l_cellType[l_linearInd]=g_cellType[g_linearInd];
        
    // // // barrier(CLK_LOCAL_MEM_FENCE);     
    
    
    // // // if(solverParams->hexLattice){
        // // // //this part is slow - essentially we do unnecessary multiple copies of data but the code is simple to uinderstand. I tried doing series of if-else statements in addition to copying faces but this always resulted in runtime-error. 
        // // // // I leave it in it for now, maybe later I can debug it better
        // // // int4 corner_offset1[26]={(int4)(1,0,0,0) , (int4)(1,1,0,0) , (int4)(0,1,0,0) , (int4)(-1,1,0,0) , (int4)(-1,0,0,0) , (int4)(-1,-1,0,0) , (int4)(0,-1,0,0) , (int4)(1,-1,0,0),
        // // // (int4)(1,0,1,0) , (int4)(1,1,1,0) , (int4)(0,1,1,0) , (int4)(-1,1,1,0) , (int4)(-1,0,1,0) , (int4)(-1,-1,1,0) , (int4)(0,-1,1,0) , (int4)(1,-1,1,0) , (int4)(0,0,1,0),
        // // // (int4)(1,0,-1,0) , (int4)(1,1,-1,0) , (int4)(0,1,-1,0) , (int4)(-1,1,-1,0) , (int4)(-1,0,-1,0) , (int4)(-1,-1,-1,0) , (int4)(0,-1,-1,0) , (int4)(1,-1,-1,0) , (int4)(0,0,-1,0)         
        // // // };     
        
        // // // for (int i = 0 ; i < 26 ; ++i){
       
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+1,0) + corner_offset1[i] ) ]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+1,0) + corner_offset1[i])];
            // // // // barrier(CLK_LOCAL_MEM_FENCE);     
        // // // }    
        
        
    // // // }else{
        // // // //copying faces
        // // // if (tx==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x,gid_y+1,gid_z+1,0))];  //central pixel                      
        // // // }
        
        // // // if (tx==l_dim_x-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+2,ty+1,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+2,gid_y+1,gid_z+1,0))];        
        // // // }
        
      
        // // // if (ty==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y,gid_z+1,0))];
        // // // }
        
      
        // // // if (ty==l_dim_y-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+2,tz+1,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+2,gid_z+1,0))];
        // // // }
      
        // // // if (tz==0){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z,0))];
        // // // }
      
        // // // if (tz==l_dim_z-1){
            // // // l_field[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_field[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
            // // // l_cellType[ext3DIndToLinear(l_dim, (int4)(tx+1,ty+1,tz+2,0))]=g_cellType[ext3DIndToLinear(g_dim, (int4)(gid_x+1,gid_y+1,gid_z+2,0))];
        // // // }
    // // // }    
    
    // // // barrier(CLK_LOCAL_MEM_FENCE);     

    
	// // // float currentConcentration=l_field[l_linearInd];
	// // // float concentrationSum=0.f;
        
    // // // // // // return;        
    
	// // // if(solverParams->hexLattice){
		// // // for(int i=0; i<solverParams->nbhdConcLen; ++i){
			// // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1,0), i, solverParams->hexLattice, nbhdConcShifts, solverParams->nbhdConcLen);
			// // // int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			// // // concentrationSum+=l_field[lInd];
		// // // }
		// // // concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
	// // // }else{
		// // // for(int i=0; i<solverParams->nbhdConcLen; ++i){
			// // // int4 shift=nbhdConcShifts[i];
			// // // int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			// // // concentrationSum+=l_field[lInd];
		// // // }
		// // // concentrationSum-=solverParams->nbhdConcLen*currentConcentration;
	// // // }
    
	
	// // // unsigned char curentCelltype=l_cellType[l_linearInd];
	// // // float currentDiffCoef=solverParams->diffCoef[curentCelltype];
	
	// // // concentrationSum*=currentDiffCoef;
	
	// // // float varDiffSumTerm=0.f;
	// // // if(solverParams->hexLattice){
		// // // for(int i=0; i<solverParams->nbhdDiffLen; ++i){
			// // // int4 shift=getShift(g_ind+(int4)(-1,-1,-1, 0), i, solverParams->hexLattice, nbhdDiffShifts, solverParams->nbhdDiffLen);
			// // // int lInd=ext3DIndToLinear(l_dim, l_ind+shift);
			// // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
		// // // }
	// // // }else{
		// // // for(int i=0; i<solverParams->nbhdDiffLen; ++i){
			// // // int lInd=ext3DIndToLinear(l_dim, l_ind+nbhdDiffShifts[i]);
			// // // varDiffSumTerm+=(solverParams->diffCoef[l_cellType[lInd]]-currentDiffCoef)*(l_field[lInd]-currentConcentration);
		// // // }
	// // // }
	    
	// // // float dx2=solverParams->dx*solverParams->dx;
    // // // float dt=solverParams->dx*solverParams->dt;
	// // // float dt_dx2=dt/dx2;

    // // // float scratch=dt_dx2*(concentrationSum+varDiffSumTerm)+(1.f-dt*solverParams->decayCoef[curentCelltype])*currentConcentration;
    
        
    
    // // // g_scratch[g_linearInd]=scratch;
            
// // // }