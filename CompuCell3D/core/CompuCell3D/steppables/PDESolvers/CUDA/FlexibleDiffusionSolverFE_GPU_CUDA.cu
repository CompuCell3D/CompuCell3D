
#include "FlexibleDiffusionSolverFE_GPU_CUDA.h"
#include "../DiffSecrData.h"

#include "CUDAUtilsHeader.h"
#include "../GPUSolverBasicData.h"
#include <iostream>
#include <Logger/CC3DLogger.h>
# define BLOCK_SIZE_FRAME (BLOCK_SIZE+2)

using std::endl;
using std::vector;
using std::string;
using std::swap;

namespace CompuCell3D {

    FlexibleDiffusionSolverFE_GPU_CUDA::FlexibleDiffusionSolverFE_GPU_CUDA() : h_solverParamPtr(NULL),
                                                                               d_field(NULL),
                                                                               d_celltype_field(NULL),
                                                                               d_boundary_field(NULL),
                                                                               d_scratch(NULL),
                                                                               d_solverParam(NULL),
                                                                               mem_size_field(0),
                                                                               mem_size_celltype_field(0) {
    }

    FlexibleDiffusionSolverFE_GPU_CUDA::~FlexibleDiffusionSolverFE_GPU_CUDA() {
        if (h_solverParamPtr)
            checkCudaErrors(cudaFreeHost(h_solverParamPtr));

        if (d_field)
            checkCudaErrors(cudaFree(d_field));

        if (d_scratch)
            checkCudaErrors(cudaFree(d_scratch));

        if (d_celltype_field)
            checkCudaErrors(cudaFree(d_celltype_field));

        if (d_boundary_field)
            checkCudaErrors(cudaFree(d_boundary_field));

    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::init(int gpuDeviceIndex, LatticeType lt, size_t fieldLen) {
        //cudaSetDevice( /*cutGetMaxGflopsDeviceId()*/0);

	//TODO: reimplement device selector
	//not the most efficient code...
	//refactoring needed (separate device selection from user messages)
	if(gpuDeviceIndex==-1){//select the fastest GPU device
        CC3D_Log(LOG_DEBUG) << "Selecting the fastest GPU device...";
		int num_devices, device;
		cudaGetDeviceCount(&num_devices);
		if (num_devices > 1) {
			  int max_multiprocessors = 0, max_device = 0;
			  for (device = 0; device < num_devices; device++) {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  if (max_multiprocessors < properties.multiProcessorCount) {
							  max_multiprocessors = properties.multiProcessorCount;
							  max_device = device;
					  }
			  }
			  cudaDeviceProp properties;
			  cudaGetDeviceProperties(&properties, max_device);
              CC3D_Log(LOG_DEBUG) << "GPU device "<<max_device<<" selected; GPU device name: "<<properties.name;
			  cudaSetDevice(max_device);
			  gpuDeviceIndex=max_device;
		}else{
            CC3D_Log(LOG_DEBUG) << "Only one GPU device available, will use it (#0)";
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, 0);
            CC3D_Log(LOG_DEBUG) << "GPU device name: "<<properties.name;
		}
	}else{
		cudaError_t err=cudaSetDevice(gpuDeviceIndex);
		if(err!=cudaSuccess){
            CC3D_Log(LOG_DEBUG) << "Can't use the GPU device # "<<gpuDeviceIndex<<" (error code: "<<err<<", err message: "<<cudaGetErrorString(err)<<")";
			exit(-1);
		}

		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, gpuDeviceIndex);
        CC3D_Log(LOG_DEBUG) << "GPU device name: "<<properties.name;
        }

        alloc(fieldLen);
    }

void FlexibleDiffusionSolverFE_GPU_CUDA::alloc(size_t fieldLen){
	unsigned int flags = cudaHostAllocMapped;
    checkCudaErrors(cudaHostAlloc((void **)&h_solverParamPtr, sizeof(SolverParams_t), flags));
    CC3D_Log(LOG_DEBUG) << "h_solverParamPtr-"<<h_solverParamPtr;


        // allocate device memory
        mem_size_field = fieldLen * sizeof(float);
        mem_size_celltype_field = fieldLen * sizeof(unsigned char);
        checkCudaErrors(cudaMalloc((void **) &d_field, mem_size_field));

        //
        checkCudaErrors(cudaMalloc((void **) &d_celltype_field, mem_size_celltype_field));

        checkCudaErrors(cudaMalloc((void **) &d_boundary_field, mem_size_celltype_field));

        //
        checkCudaErrors(cudaMalloc((void **) &d_scratch, mem_size_field));

        //enabling sharing of the h_solverParamPtr between host and device


        checkCudaErrors(cudaHostGetDevicePointer((void **) &d_solverParam, (void *) h_solverParamPtr, 0));
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::prepareSolverParams(Dim3D fieldDim, DiffusionData const &diffData) {
        SolverParams_t &h_solverParam = *h_solverParamPtr;
        h_solverParam.dimx = fieldDim.x;
        h_solverParam.dimy = fieldDim.y;
        h_solverParam.dimz = fieldDim.z;

    h_solverParam.dx=1.0;
    h_solverParam.dt=1.0;
    h_solverParam.numberOfCelltypes=2;
	
	for (int i=0 ; i<UCHAR_MAX+1 ; ++i){
		h_solverParam.diffCoef[i]=diffData.diffCoef[i];
		h_solverParam.decayCoef[i]=diffData.decayCoef[i];
        CC3D_Log(LOG_TRACE) << "h_solverParam.diffCoef["<<i<<"]="<<h_solverParam.diffCoef[i];
        }
    }

string FlexibleDiffusionSolverFE_GPU_CUDA::solverName(){
    CC3D_Log(LOG_DEBUG) << "Calling FlexibleDiffusionSolverFE_GPU_CUDA::solverName";
        return "FlexibleDiffusionSolverFE_CUDA";
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::fieldHostToDevice(float const *h_field) {
        checkCudaErrors(cudaMemcpy(d_field, h_field, mem_size_field,
                                   cudaMemcpyHostToDevice));
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::fieldDeviceToHost(float *h_field) const {
        checkCudaErrors(cudaMemcpy(h_field, d_scratch, mem_size_field, cudaMemcpyDeviceToHost));
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::swapScratchAndField() {
        swap(d_field, d_scratch);
    }

void FlexibleDiffusionSolverFE_GPU_CUDA::initCellTypeArray(unsigned char *arr, size_t arrLength){
    CC3D_Log(LOG_TRACE) << "h_celltype_field->getArraySize()="<<arrLength<<" mem_size_celltype_field="<<mem_size_celltype_field;
        ////h_celltype_field=cellTypeMonitorPlugin->getCellTypeArray();
        checkCudaErrors(
                cudaMemcpy(d_celltype_field, arr, arrLength * sizeof(*d_celltype_field), cudaMemcpyHostToDevice));
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::initBoundaryArray(unsigned char *arr, size_t arrLength) {
        checkCudaErrors(
                cudaMemcpy(d_boundary_field, arr, arrLength * sizeof(*d_boundary_field), cudaMemcpyHostToDevice));
    }


    __global__ void
    diffSolverKernel(float *field, float *scratch, unsigned char *celltype, SolverParams_t *solverParams) {
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;

        int bz = 0; //simulated blockIdx.z
        int DIMX = solverParams->dimx;
        int DIMY = solverParams->dimy;
        int DIMZ = solverParams->dimz;

        int bz_max = DIMZ / BLOCK_SIZE;

        //each thread copies data into shared memory
        int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;

        __shared__ float fieldBlock[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        __shared__ unsigned char celltypeBlock[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        __shared__ float scratchBlock[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];


        for (bz = 0; bz < bz_max; ++bz) {


            //mapping from block,threadIdx to x,y,zof the inner frame
            int x = bx * BLOCK_SIZE + tx;
            int y = by * BLOCK_SIZE + ty;
            int z = bz * BLOCK_SIZE + tz;

            //int offset=threadsPerBlock*bx+threadsPerBlock*blockDim.x*by+DIMX*DIMY*BLOCK_SIZE*bz;

            fieldBlock[tx + 1][ty + 1][tz + 1] = field[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) + x +
                                                       1];
            celltypeBlock[tx + 1][ty + 1][tz + 1] = celltype[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) +
                                                             x + 1];

            scratchBlock[tx][ty][tz] = 0.0;

            //fieldBlock(tx+1, ty+1, tz+1) = field[offset+tz*BLOCK_SIZE*BLOCK_SIZE+ty*BLOCK_SIZE+tx];
            if (tx == 0) {
                fieldBlock[0][ty + 1][tz + 1] = field[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) + x];
                celltypeBlock[0][ty + 1][tz + 1] = celltype[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) +
                                                            x];
            }

            if (tx == BLOCK_SIZE - 1) {
                fieldBlock[BLOCK_SIZE + 1][ty + 1][tz + 1] = field[(z + 1) * (DIMX + 2) * (DIMY + 2) +
                                                                   (y + 1) * (DIMX + 2) + x + 2];
                celltypeBlock[BLOCK_SIZE + 1][ty + 1][tz + 1] = celltype[(z + 1) * (DIMX + 2) * (DIMY + 2) +
                                                                         (y + 1) * (DIMX + 2) + x + 2];
            }

            if (ty == 0) {
                fieldBlock[tx + 1][0][tz + 1] = field[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y) * (DIMX + 2) + x + 1];
                celltypeBlock[tx + 1][0][tz + 1] = celltype[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y) * (DIMX + 2) + x +
                                                            1];
            }

            if (ty == BLOCK_SIZE - 1) {
                fieldBlock[tx + 1][BLOCK_SIZE + 1][tz + 1] = field[(z + 1) * (DIMX + 2) * (DIMY + 2) +
                                                                   (y + 2) * (DIMX + 2) + x + 1];
                celltypeBlock[tx + 1][BLOCK_SIZE + 1][tz + 1] = celltype[(z + 1) * (DIMX + 2) * (DIMY + 2) +
                                                                         (y + 2) * (DIMX + 2) + x + 1];
            }

            if (tz == 0) {
                fieldBlock[tx + 1][ty + 1][0] = field[(z) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) + x + 1];
                celltypeBlock[tx + 1][ty + 1][0] = celltype[(z) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) + x +
                                                            1];
            }

            if (tz == BLOCK_SIZE - 1) {
                fieldBlock[tx + 1][ty + 1][BLOCK_SIZE + 1] = field[(z + 2) * (DIMX + 2) * (DIMY + 2) +
                                                                   (y + 1) * (DIMX + 2) + x + 1];
                celltypeBlock[tx + 1][ty + 1][BLOCK_SIZE + 1] = celltype[(z + 2) * (DIMX + 2) * (DIMY + 2) +
                                                                         (y + 1) * (DIMX + 2) + x + 1];
            }


            __syncthreads();

            //solve actual diff equation
            float concentrationSum = 0.0;
            float dt_dx2 = solverParams->dt / (solverParams->dx * solverParams->dx);

            int curentCelltype = celltypeBlock[tx + 1][ty + 1][tz + 1];

            concentrationSum = fieldBlock[tx + 2][ty + 1][tz + 1] + fieldBlock[tx + 1][ty + 2][tz + 1] +
                               fieldBlock[tx + 1][ty + 1][tz + 2]
                               + fieldBlock[tx][ty + 1][tz + 1] + fieldBlock[tx + 1][ty][tz + 1] +
                               fieldBlock[tx + 1][ty + 1][tz] - 6 * fieldBlock[tx + 1][ty + 1][tz + 1];

            float *diffCoef = solverParams->diffCoef;
            float *decayCoef = solverParams->decayCoef;


            concentrationSum *= diffCoef[curentCelltype];


            float varDiffSumTerm = 0.0;

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
            varDiffSumTerm += (diffCoef[celltypeBlock[tx + 2][ty + 1][tz + 1]] - diffCoef[curentCelltype]) *
                              (fieldBlock[tx + 2][ty + 1][tz + 1] - fieldBlock[tx + 1][ty + 1][tz + 1]);
            //y partial derivatives
            varDiffSumTerm += (diffCoef[celltypeBlock[tx + 1][ty + 2][tz + 1]] - diffCoef[curentCelltype]) *
                              (fieldBlock[tx + 1][ty + 2][tz + 1] - fieldBlock[tx + 1][ty + 1][tz + 1]);
            //z partial derivatives
            varDiffSumTerm += (diffCoef[celltypeBlock[tx + 1][ty + 1][tz + 2]] - diffCoef[curentCelltype]) *
                              (fieldBlock[tx + 1][ty + 1][tz + 2] - fieldBlock[tx + 1][ty + 1][tz + 1]);


            //OK
            scratchBlock[tx][ty][tz] = dt_dx2 * (concentrationSum + varDiffSumTerm) +
                                       (1 - solverParams->dt * decayCoef[curentCelltype]) *
                                       fieldBlock[tx + 1][ty + 1][tz + 1];



            //simple consistency check
            //scratchBlock[tx][ty][tz]=concentrationSum;
            //scratchBlock[tx][ty][tz]=fieldBlock[tx+2][ty+1][tz+1]+fieldBlock[tx][ty+1][tz+1]+fieldBlock[tx+1][ty+2][tz+1]+fieldBlock[tx+1][ty][tz+1]+fieldBlock[tx+1][ty+1][tz+2]+fieldBlock[tx+1][ty+1][tz];

            //scratchBlock[tx][ty][tz]=fieldBlock[tx+1][ty+1][tz+1];

            //fieldBlock[tx+1][ty+1][tz+1]=3000.0f;
            __syncthreads();

            //copy scratchBlock to scratch field on the device

            scratch[(z + 1) * (DIMX + 2) * (DIMY + 2) + (y + 1) * (DIMX + 2) + x + 1] = scratchBlock[tx][ty][tz];
            //scratch[(z+1)*(DIMX+2)*(DIMY+2)+(y+1)*(DIMX+2)+x+1]=3000.0;

            __syncthreads();

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
        //__syncthreads();
    }

    void FlexibleDiffusionSolverFE_GPU_CUDA::diffuseSingleField() {
        //we cannot access device variable (e.g. d_solverParam) from this part of the code - only kernel is allowed to do this
        //here we are using page-locked memory to share SolverParams_t structure between device and host
        unsigned int dimX = h_solverParamPtr->dimx;
        unsigned int dimY = h_solverParamPtr->dimy;
        unsigned int dimZ = h_solverParamPtr->dimz;

        SolverParams_t *d_solverParamFromMappedMemory;
        cudaHostGetDevicePointer((void **) &d_solverParamFromMappedMemory, (void *) h_solverParamPtr, 0);

        //cutilSafeCall(cudaMemcpy(d_solverParamFromMappedMemory, h_solverParam, sizeof(SolverParams_t ),cudaMemcpyHostToDevice) );

        // setup execution parameters
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(dimX / threads.x, dimY / threads.y);

        diffSolverKernel<<< grid, threads >>>(d_field, d_scratch, d_celltype_field, d_solverParamFromMappedMemory);
        //diffSolverKernel<<< grid, threads >>>(d_field, d_scratch,d_celltype_field,d_solverParam);
        cudaThreadSynchronize();//TODO: this synchronization looks redundant. Copying memory back to host implies implicit synchronization
    }


}//namespace CompuCell3D