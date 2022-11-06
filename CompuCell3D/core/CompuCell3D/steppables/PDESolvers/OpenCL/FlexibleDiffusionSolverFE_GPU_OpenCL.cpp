#include "FlexibleDiffusionSolverFE_GPU_OpenCL.h"
#include "../DiffSecrData.h"
//#include "BasicUtils/BasicException.h"

#include "../GPUSolverBasicData.h"
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include "OpenCLHelper.h"
#include <Logger/CC3DLogger.h>
//#include <algorithm>//TODO: remove
//#include <functional>//TODO: remove
//#include <limits>//TODO: remove

//depricated

# define BLOCK_SIZE_FRAME (BLOCK_SIZE+2)

using std::endl;
using std::vector;
using std::string;
using std::swap;
using std::numeric_limits;

namespace CompuCell3D {


    FlexibleDiffusionSolverFE_GPU_OpenCL::FlexibleDiffusionSolverFE_GPU_OpenCL() :
            oclHelper(NULL),
            h_solverParamPtr(NULL),
            program(NULL),
            kernel(NULL),
            d_field(NULL),
            d_celltype_field(NULL),
            d_boundary_field(NULL),
            d_scratch(NULL),
            d_solverParam(NULL) {
    }

    FlexibleDiffusionSolverFE_GPU_OpenCL::~FlexibleDiffusionSolverFE_GPU_OpenCL() {
        //TODO: should wait here for the end of the GPU operations somehow...
        CC3D_Log(LOG_DEBUG) << "FlexibleDiffusionSolverFE_GPU_OpenCL: destroying GPU objects...";
	oclHelper->Finish();

        cl_int res;

        if (d_field) {
            res = clReleaseMemObject(d_field);
            assert(res == CL_SUCCESS);
        }

        if (d_scratch) {
            res = clReleaseMemObject(d_scratch);
            assert(res == CL_SUCCESS);
        }

        if (d_celltype_field) {
            res = clReleaseMemObject(d_celltype_field);
            assert(res == CL_SUCCESS);
        }

        if (d_solverParam) {
            res = clReleaseMemObject(d_solverParam);
            assert(res == CL_SUCCESS);
        }

        if (d_boundary_field) {
            res = clReleaseMemObject(d_boundary_field);
            assert(res == CL_SUCCESS);
        }

        res = clReleaseKernel(kernel);
        assert(res == CL_SUCCESS);

        assert(res == CL_SUCCESS);

        delete h_solverParamPtr;

	delete oclHelper;
	CC3D_Log(LOG_DEBUG) << "FlexibleDiffusionSolverFE_GPU_OpenCL: destroying GPU objects... finished";
	
}

void FlexibleDiffusionSolverFE_GPU_OpenCL::init(int gpuDeviceIndex, LatticeType lt, size_t fieldLen) {
    CC3D_Log(LOG_DEBUG) << "Initialize OpenCL object and context";
    //setup devices and context

    latticeType=lt;
    if(latticeType==HEXAGONAL_LATTICE) {
        CC3D_Log(LOG_DEBUG) << "Using hexagonal lattice"; }
    else {
        CC3D_Log(LOG_DEBUG) << "Using square lattice";
    };
    
	oclHelper=new OpenCLHelper(gpuDeviceIndex);

        alloc(fieldLen);

    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::alloc(size_t fieldLen) {

        h_solverParamPtr = new SolverParams_t();

	// allocate device memory
	field_len=fieldLen;
    size_t mem_size_field=fieldLen*sizeof(float);
	size_t mem_size_celltype_field=fieldLen*sizeof(unsigned char);
	CC3D_Log(LOG_DEBUG) << "Initializing GPU memory";
        d_field = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_field);
        if (!d_field) {
            CC3D_Log(LOG_DEBUG) << "Can't allocate memory for d_field";
            exit(1);
        }

        d_scratch = oclHelper->CreateBuffer(CL_MEM_WRITE_ONLY, mem_size_field);
        if (!d_scratch) {
            CC3D_Log(LOG_DEBUG) << "Can't allocate memory for d_scratch: ";
            exit(1);
        }

        d_celltype_field = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_celltype_field);
        if (!d_celltype_field) {
            CC3D_Log(LOG_DEBUG) << "Can't allocate memory for d_celltype_field";
		exit(1);
	}

        d_boundary_field = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_celltype_field);
        if (!d_boundary_field) {
            CC3D_Log(LOG_DEBUG) << "Can't allocate memory for d_boundary_field";
            exit(1);
        }

        d_solverParam = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(SolverParams));
        if (!d_solverParam) {
            CC3D_Log(LOG_DEBUG) << "Can't allocate memory for d_solverParam: "; exit(1);
	}
	CC3D_Log(LOG_DEBUG) << "building OpenCL program";

        const char *kernelSource[] = {"lib/CompuCell3DSteppables/OpenCL/GPUSolverParams.h",
                                      "lib/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl"};

        if (!oclHelper->LoadProgram(kernelSource, 2, program)) {
            //ASSERT_OR_THROW("Can't load the OpenCL kernel", false);
            CC3D_Log(LOG_DEBUG) << "Can't load the OpenCL kernel";exit(-1);
	}

        CreateKernel();

    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::prepareSolverParams(Dim3D fieldDim, DiffusionData const &diffData) {
        SolverParams_t &h_solverParam = *h_solverParamPtr;
        h_solverParam.dimx = fieldDim.x;
        h_solverParam.dimy = fieldDim.y;
        h_solverParam.dimz = fieldDim.z;

        h_solverParam.dx2 = 1.0;
        h_solverParam.dt = 1.0;
        h_solverParam.numberOfCelltypes = 2;

        for (int i = 0; i < UCHAR_MAX + 1; ++i) {
            h_solverParam.diffCoef[i] = diffData.diffCoef[i];
            h_solverParam.decayCoef[i] = diffData.decayCoef[i];
            CC3D_Log(LOG_TRACE) << "h_solverParam.diffCoef["<<i<<"]="<<h_solverParam.diffCoef[i];
        }

        oclHelper->WriteBuffer(d_solverParam, h_solverParamPtr, 1);

        //TODO: I'd better move it out of here... Along with once-per simulation set of parameters
        localWorkSize[0] = BLOCK_SIZE;
        localWorkSize[1] = BLOCK_SIZE;
        //TODO: BLOCK size can be non-optimal in terms of maximum performance
        localWorkSize[2] = std::min((unsigned int) (oclHelper->getMaxWorkGroupSize() / (BLOCK_SIZE * BLOCK_SIZE)),
                                    h_solverParam.dimz);

        static int i = 0;//just to avoid output polluting
        if (i == 0) {
            ++i;
            CC3D_Log(LOG_DEBUG) << "Block size is: "<<localWorkSize[0]<<"x"<<localWorkSize[1]<<"x"<<localWorkSize[2];
        }
    }

    string FlexibleDiffusionSolverFE_GPU_OpenCL::solverName() {
        return "FlexibleDiffusionSolverFE_OpenCL";
    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::fieldHostToDevice(float const *h_field) {
        assert(oclHelper);
        oclHelper->WriteBuffer(d_field, h_field, field_len);
    }

//for debugging
    void CheckConcentrationField(SolverParams const *h_solverParamPtr, float const *h_field) {
        //size_t lim=(h_solverParamPtr->dimx+2)*(h_solverParamPtr->dimy+2)*(h_solverParamPtr->dimz+2);

        //for(size_t i=0; i<lim; ++i){
        //	h_field[i]=2.f;
        //}
        CC3D_Log(LOG_TRACE) << h_field[800];
	double sum=0.f;
	float minVal=numeric_limits<float>::max();
	float maxVal=-numeric_limits<float>::max();
	for(unsigned int z=1; z<=h_solverParamPtr->dimz; ++z){
		for(unsigned int y=1; y<=h_solverParamPtr->dimy; ++y){
			for(unsigned int x=1; x<=h_solverParamPtr->dimx; ++x){
				float val=h_field[z*(h_solverParamPtr->dimx+2)*(h_solverParamPtr->dimy+2)+y*(h_solverParamPtr->dimx+2)+x];
				sum+=val;
				minVal=std::min(val, minVal);
				maxVal=std::max(val, maxVal);
			}
		}
	}
	CC3D_Log(LOG_DEBUG) << "min: "<<minVal<<"; max: "<<maxVal<<" "<<sum;
    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::fieldDeviceToHost(float *h_field) const {
        assert(oclHelper);
        if (oclHelper->ReadBuffer(d_scratch, h_field, field_len) != CL_SUCCESS) {
            CC3D_Log(LOG_DEBUG) << "Error on reading"; exit(-1);
	}

        //TODO: disable code
        //CheckConcentrationField(h_solverParamPtr, h_field);

    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::swapScratchAndField() {
        swap(d_field, d_scratch);
    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::initCellTypeArray(unsigned char *arr, size_t arrLength) {
        assert(oclHelper);
        oclHelper->WriteBuffer(d_celltype_field, arr, arrLength);
    }

    void FlexibleDiffusionSolverFE_GPU_OpenCL::initBoundaryArray(unsigned char *arr, size_t arrLength) {
        assert(oclHelper);
        oclHelper->WriteBuffer(d_boundary_field, arr, arrLength);
    }


    void FlexibleDiffusionSolverFE_GPU_OpenCL::diffuseSingleField() {

        //TODO: check if all the parameters should be copied to the GPU device every time
        PrepareKernelParams();

        cl_int err;
        if (latticeType == SQUARE_LATTICE) {
            const size_t globalWorkSize[] = {h_solverParamPtr->dimx, h_solverParamPtr->dimy, h_solverParamPtr->dimz};
            CC3D_Log(LOG_TRACE) << "Block size is: "<<localWorkSize[0]<<"x"<<localWorkSize[1]<<"x"<<localWorkSize[2]<<
			"; globalWorkSize is: "<<globalWorkSize[0]<<"x"<<globalWorkSize[1]<<"x"<<globalWorkSize[2];
//            	" "<<maxWorkGroupSize;
            //execute the kernel
            err = oclHelper->EnqueueNDRangeKernel(kernel, 3, globalWorkSize, localWorkSize);
        } else {
            CC3D_Log(LOG_TRACE) << "Running the hex kernel";
		const size_t globalWorkSize[]={h_solverParamPtr->dimx, h_solverParamPtr->dimy};
		CC3D_Log(LOG_TRACE) << "Block size is: "<<localWorkSize[0]<<"x"<<localWorkSize[1]<<
			"; globalWorkSize is: "<<globalWorkSize[0]<<"x"<<globalWorkSize[1];
            //execute the kernel
            err = oclHelper->EnqueueNDRangeKernel(kernel, 2, globalWorkSize, localWorkSize);
        }
        if (err != CL_SUCCESS) {
            CC3D_Log(LOG_DEBUG) << "clEnqueueNDRangeKernel: " << oclHelper->ErrorString(err);
            throw std::runtime_error("OpenCL kernel run failed");
        }
    }


////////////////////
//helper functions//
////////////////////


    void FlexibleDiffusionSolverFE_GPU_OpenCL::CreateKernel() {
        //initialize our kernel from the program
        cl_int err;
        string kernelName;
        if (latticeType == SQUARE_LATTICE) {
            kernelName = "diff3D";
        } else {
            kernelName = "diff2DHex";
        }
        kernel = clCreateKernel(program, kernelName.c_str(), &err);
        CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel " << kernelName.c_str() << ": " << oclHelper->ErrorString(err);
    }


    void FlexibleDiffusionSolverFE_GPU_OpenCL::PrepareKernelParams() {
        cl_int err = CL_SUCCESS;
        if (latticeType == SQUARE_LATTICE) {
            //set the arguements of our kernel
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_field);
            err = err | clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_celltype_field);
            err = err | clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_scratch);
            err = err | clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_solverParam);
            //local (shared) memory arrays
            err = err | clSetKernelArg(kernel, 4, sizeof(float) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) *
                                                  (localWorkSize[2] + 2), NULL);
            err = err | clSetKernelArg(kernel, 5,
                                       sizeof(unsigned char) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) *
                                       (localWorkSize[2] + 2), NULL);
            err = err |
                  clSetKernelArg(kernel, 6, sizeof(float) * localWorkSize[0] * localWorkSize[1] * localWorkSize[2],
                                 NULL);

        } else {
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_field);
            err = err | clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_celltype_field);
            err = err | clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_scratch);
            err = err | clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_solverParam);
            err = err |
                  clSetKernelArg(kernel, 4, sizeof(float) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2), NULL);
            err = err |
                  clSetKernelArg(kernel, 5, sizeof(unsigned char) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2),
                                 NULL);
            err = err | clSetKernelArg(kernel, 6, sizeof(float) * localWorkSize[0] * localWorkSize[1], NULL);
        }


        if (err != CL_SUCCESS) {
            CC3D_Log(LOG_DEBUG) << "FlexibleDiffusionSolverFE_GPU_OpenCL::PrepareKernelParams: " << oclHelper->ErrorString(err);
            throw std::runtime_error("Kernel preparing error");
        }
        
    }


}//namespace CompuCell3D