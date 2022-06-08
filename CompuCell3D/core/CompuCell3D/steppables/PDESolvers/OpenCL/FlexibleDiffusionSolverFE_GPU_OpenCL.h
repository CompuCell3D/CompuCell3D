#ifndef COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_OPENCL_H
#define COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_OPENCL_H

#include "../FlexibleDiffusionSolverFE_GPU_Device.h"

// 2012 Mitja:
// #include <CL/opencl.h>	
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif


//depricated

struct SolverParams;

namespace CompuCell3D {

    class OpenCLHelper;

    class PDESOLVERSGPU_EXPORT FlexibleDiffusionSolverFE_GPU_OpenCL : public FlexibleDiffusionSolverFE_GPU_Device {
        SolverParams *h_solverParamPtr;

        cl_mem d_field;
        cl_mem d_celltype_field;
        cl_mem d_boundary_field;
        cl_mem d_scratch;
        cl_mem d_solverParam;

        cl_program program;
        cl_kernel kernel;

        size_t field_len;
        size_t localWorkSize[3];//block size

        OpenCLHelper *oclHelper;
//	size_t mem_size_celltype_field;

    public:
        FlexibleDiffusionSolverFE_GPU_OpenCL();

        ~FlexibleDiffusionSolverFE_GPU_OpenCL();

    protected:
        virtual void alloc(size_t fieldLen);

    public:
        virtual void init(int gpuDeviceIndex, LatticeType lt, size_t fieldLen);

        virtual void prepareSolverParams(Dim3D fieldDim, DiffusionData const &diffData);

        virtual std::string solverName();

        virtual void fieldHostToDevice(float const *h_field);

        virtual void fieldDeviceToHost(float *h_field) const;

        virtual void diffuseSingleField();

        virtual void swapScratchAndField();

        virtual void initCellTypeArray(unsigned char *arr, size_t arrLength);

        virtual void initBoundaryArray(unsigned char *arr, size_t arrLength);

    private:

        //make a kernel and pass the parameters there
        void CreateKernel();

        //pass parameters to the kernel
        void PrepareKernelParams();
    };

}//namespace CompuCell3D

#endif