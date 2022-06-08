#ifndef COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_CUDA_H
#define COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_CUDA_H

#include "../FlexibleDiffusionSolverFE_GPU_Device.h"

struct SolverParams;

namespace CompuCell3D {

    class PDESOLVERSGPU_EXPORT FlexibleDiffusionSolverFE_GPU_CUDA : public FlexibleDiffusionSolverFE_GPU_Device {
        SolverParams *h_solverParamPtr;

        //device
        float *d_field;
        unsigned char *d_celltype_field;
        unsigned char *d_boundary_field;
        float *d_scratch;
        SolverParams *d_solverParam;

        int mem_size_field;
        int mem_size_celltype_field;

    public:
        FlexibleDiffusionSolverFE_GPU_CUDA();

        ~FlexibleDiffusionSolverFE_GPU_CUDA();

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
    };

}//namespace CompuCell3D

#endif