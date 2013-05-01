


//stencil-like diffusion operator
//dc_dx2 stands for diffusion coefficient devided by dx^2
float diffOpGlob(int4 ind, int3 dim, __global int4 const *nbhdShifts, int nbhdLen, float dc_dx2, float currField, __global float const *field)
{
	float scratch=0.f;

	for(int i=0; i<nbhdLen; ++i){
		int4 shift=nbhdShifts[i];
		int4 shiftedInd=ind+shift;
		int nbhdInd=d3To1d(shiftedInd, dim);
		scratch+=field[nbhdInd];
	}
	scratch-=nbhdLen*currField;
		
	return scratch*dc_dx2;
	//return scratch;
}


//producing matrix-vewctor product for (nextField-deltaT*diffOp(nextField)),
//non-boundary part only
__kernel void ImplicitMatrixProdCore(float dt,
	__global UniSolverParams_t  const *solverParams,
	__global const unsigned char * g_cellType,
	__global const float * g_field,
	__global float * g_result) 
{
	int4 ind3d={get_global_id(0), get_global_id(1), get_global_id(2), 0};
	int3 dim={solverParams->xDim, solverParams->yDim, solverParams->zDim};

	if(ind3d.x>=dim.x||ind3d.y>=dim.y||ind3d.z>=dim.z)
		return;

	const bool isOnBoundary=(ind3d.x==0||ind3d.x==dim.x-1||
		ind3d.y==0||ind3d.y==dim.y-1||
		((ind3d.z==0||ind3d.z==dim.z-1)&&dim.z!=1));

	float dx2=solverParams->dx*solverParams->dx;

	size_t ind=d3To1d(ind3d, dim);
	unsigned char cellType=g_cellType[ind];
	float dc_dx2=solverParams->diffCoef[cellType]/dx2;

	float currField=g_field[ind];

	if(!isOnBoundary){
		float res=currField-dt*diffOpGlob(ind3d, dim, solverParams->nbhdShifts, solverParams->nbhdConcLen, dc_dx2, currField, g_field);
		g_result[ind]=res;
	}
};


float diffOpGlobPeriodic(int4 ind, int3 dim, __global int4 const *nbhdShifts, int nbhdLenOuter, float dc_dx2, float currField, __global float const *field,
	int3 isOnBoundary)
{
	float scratch=0.f;

	int nbhdLen=0;
	for(int i=0; i<nbhdLenOuter; ++i){
		int4 shift=nbhdShifts[i];

		if(isOnBoundary.x&&shift.x!=0)
			continue;
		if(isOnBoundary.y&&shift.y!=0)
			continue;
		if(isOnBoundary.z&&shift.z!=0)
			continue;
		++nbhdLen;
		int4 shiftedInd=ind+shift;
		//clamping periodic
		{
			if(shiftedInd.x<0)
				shiftedInd.x=dim.x-1;
			else if(shiftedInd.x>=dim.x)
				shiftedInd.x=0;
			if(shiftedInd.y<0)
				shiftedInd.y=dim.y-1;
			else if(shiftedInd.y>=dim.y)
				shiftedInd.y=0;
			if(shiftedInd.z<0)
				shiftedInd.z=dim.z-1;
			else if(shiftedInd.z>=dim.z)
				shiftedInd.z=0;
		}
		int nbhdInd=d3To1d(shiftedInd, dim);
		scratch+=field[nbhdInd];
	}
	scratch-=nbhdLen*currField;
		
	return scratch*dc_dx2;
	//return scratch;
}

float processBoundary(float dt, float dx2, int ind, __global UniSolverParams_t  const *solverParams, int4 ind3d, int3 dim, int3 isOnBoundary[2], int4 internalNbhdInd, BCPosition bcPos, 
	__global const unsigned char * g_cellType, __global GPUBoundaryConditions_t  const *boundaryConditions, __global float const *g_field/*, bool *isConstValue*/){

		
		float currField=g_field[ind];
		unsigned char cellType=g_cellType[ind];
		float dc_dx2=solverParams->diffCoef[cellType]/dx2;

		BCType bcType=boundaryConditions->planePositions[bcPos];

		float res;

		if(bcType==BC_CONSTANT_VALUE){
			//*isConstValue=true;
			//res=boundaryConditions->values[bcPos];
			res=currField;//initial rhs should be modified to boundaryConditions->values[bcPos] for these members
		}else if(bcType==BC_CONSTANT_DERIVATIVE){
			//normal stencilOp along other axes
			//float currValue=currField-2*solverParams->diffCoef[cellType]/solverParams->dx*dt*boundaryConditions->values[bcPos];
			//initial rhs should be modified to boundaryConditions->values[bcPos] for these members
			res=currField-dt*diffOpGlobPeriodic(ind3d, dim, solverParams->nbhdShifts, solverParams->nbhdConcLen, dc_dx2, currField, g_field, isOnBoundary[1]);

			int nbhdInd=d3To1d(internalNbhdInd, dim);
			res-=2*dc_dx2*dt*g_field[nbhdInd];
		} else{//assume periodic BC
			res=currField-dt*diffOpGlobPeriodic(ind3d, dim, solverParams->nbhdShifts, solverParams->nbhdConcLen, dc_dx2, currField, g_field,	isOnBoundary[0]);
		}
		return res;
}

//producing matrix-vewctor product for (nextField-deltaT*diffOp(nextField)),
//boundaries only
__kernel void ImplicitMatrixProdBoundaries(float dt,
	__global UniSolverParams_t  const *solverParams,
	__global const unsigned char * g_cellType,
	__global const float * g_field,
	__global GPUBoundaryConditions_t  const *boundaryConditions,
	__global float * g_result)
{
	int3 dim={solverParams->xDim, solverParams->yDim, solverParams->zDim};
		
	//bool isConstValue=false;

	float dx2=solverParams->dx*solverParams->dx;

	int4 isOnBoundary[2];
		
	//along x axis
	//map 0 direction to y and 1st to z
	if(get_global_id(0)<dim.y&&get_global_id(1)<dim.z){

		bool isOnYBoundary=(get_global_id(0)==0||get_global_id(0)==dim.y-1);
		bool isOnZBoundary=(get_global_id(1)==0||get_global_id(1)==dim.z-1);
		
		isOnBoundary[0].x=false; isOnBoundary[0].y=isOnYBoundary; isOnBoundary[0].z=isOnZBoundary;
		isOnBoundary[1].x=true;  isOnBoundary[1].y=isOnYBoundary; isOnBoundary[1].z=isOnZBoundary;

		{//BC_MIN_X
			int4 ind3d={0, get_global_id(0), get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.x+=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MIN_X, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}

		{//BC_MAX_X
			int4 ind3d={dim.x-1, get_global_id(0), get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.x-=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MAX_X, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}
	}
		
	//along y axis, top and bottom
	//map 0 direction to x and 1st to z
	if(get_global_id(0)<dim.x&&get_global_id(1)<dim.z){

		bool isOnXBoundary=(get_global_id(0)==0||get_global_id(0)==dim.x-1);
		bool isOnZBoundary=(get_global_id(1)==0||get_global_id(1)==dim.z-1);

		isOnBoundary[0].x=isOnXBoundary; isOnBoundary[0].y=false; isOnBoundary[0].z=isOnZBoundary;
		isOnBoundary[1].x=isOnXBoundary; isOnBoundary[1].y=true;  isOnBoundary[1].z=isOnZBoundary;

		{//BC_MIN_Y
			int4 ind3d={get_global_id(0), 0, get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.y+=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MIN_Y, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}

		{//BC_MAX_Y
			int4 ind3d={get_global_id(0), dim.y-1, get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.y-=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MAX_Y, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}
		
	}

	if(dim.z==1)//2D case
		return;

	//along z axis
	//map 0 direction to x and 1st to y
	if(get_global_id(0)<dim.x&&get_global_id(1)<dim.y){

		bool isOnXBoundary=(get_global_id(0)==0||get_global_id(0)==dim.x-1);
		bool isOnYBoundary=(get_global_id(1)==0||get_global_id(1)==dim.y-1);
		
		isOnBoundary[0].x=isOnXBoundary; isOnBoundary[0].y=isOnYBoundary;  isOnBoundary[0].z=false;
		isOnBoundary[1].x=isOnXBoundary; isOnBoundary[1].y=isOnYBoundary;  isOnBoundary[1].z=true;

		{//BC_MIN_Z
			int4 ind3d={get_global_id(0), get_global_id(1), 0, 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.z+=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MIN_Z, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}

		{//BC_MAX_Z
			int4 ind3d={get_global_id(0), get_global_id(1), dim.z-1, 0};
			size_t ind=d3To1d(ind3d, dim);
			int4 nbhdInd3d=ind3d; nbhdInd3d.z-=1;

			g_result[ind]=processBoundary(dt, dx2, ind, solverParams, ind3d, dim, isOnBoundary, nbhdInd3d, BC_MAX_Z, g_cellType, boundaryConditions, g_field/*, &isConstValue*/);
		}
			
	}
}


void ModifyRhsAccordingToBC(int ind, float dt,  __global UniSolverParams_t  const *solverParams, 
	__global GPUBoundaryConditions_t  const *boundaryConditions,
	BCPosition bcPos, unsigned char cellType, __global float * g_rhs)
{
	BCType bcType=boundaryConditions->planePositions[bcPos];

	float res;

	if(bcType==BC_CONSTANT_VALUE){
		res=boundaryConditions->values[bcPos];
	}else if(bcType==BC_CONSTANT_DERIVATIVE){
		res=g_rhs[ind]-2*solverParams->diffCoef[cellType]/solverParams->dx*dt*boundaryConditions->values[bcPos];
	}//do nothing for periodic
	g_rhs[ind]=res;
}

//should be invoked only if non-periodic boundary conditions present
__kernel void ApplyBCToRHS(float dt,
	__global UniSolverParams_t  const *solverParams,
	__global const unsigned char * g_cellType,
	__global float * g_field,
	__global GPUBoundaryConditions_t  const *boundaryConditions)
{
	int3 dim={solverParams->xDim, solverParams->yDim, solverParams->zDim};
		
	//bool isConstValue=false;

	float dx2=solverParams->dx*solverParams->dx;

	//along x axis
	//map 0 direction to y and 1st to z
	if(get_global_id(0)<dim.y&&get_global_id(1)<dim.z){

		{//BC_MIN_X
			int4 ind3d={0, get_global_id(0), get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MIN_X, cellType, g_field);
		}

		{//BC_MAX_X
			int4 ind3d={dim.x-1, get_global_id(0), get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MAX_X, cellType, g_field);
		}
	}
		
	//along y axis, top and bottom
	//map 0 direction to x and 1st to z
	if(get_global_id(0)<dim.x&&get_global_id(1)<dim.z){
		
		{//BC_MIN_Y
			int4 ind3d={get_global_id(0), 0, get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MIN_Y, cellType, g_field);
		}

		{//BC_MAX_Y
			int4 ind3d={get_global_id(0), dim.y-1, get_global_id(1), 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MAX_Y, cellType, g_field);
		}
		
	}
	
	//along z axis
	//map 0 direction to x and 1st to y
	if(get_global_id(0)<dim.x&&get_global_id(1)<dim.y){

		{//BC_MIN_Z
			int4 ind3d={get_global_id(0), get_global_id(1), 0, 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MIN_Z, cellType, g_field);
		}

		{//BC_MAX_Z
			int4 ind3d={get_global_id(0), get_global_id(1), dim.z-1, 0};
			size_t ind=d3To1d(ind3d, dim);
			unsigned char cellType=g_cellType[ind];
			ModifyRhsAccordingToBC(ind, dt, solverParams, boundaryConditions, BC_MAX_Z, cellType, g_field);
		}
	}
}