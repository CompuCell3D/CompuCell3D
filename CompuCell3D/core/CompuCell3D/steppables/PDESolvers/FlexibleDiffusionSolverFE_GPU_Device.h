#ifndef COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_DEVICE_H
#define COMPUCELL3DFLEXIBLEDIFFUSIONSOLVERFE_GPU_DEVICE_H

#include "PDESolversGPUDllSpecifier.h"
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>
#include <string>
//#include "DiffSecrData.h"

namespace CompuCell3D {

    class DiffusionData;

    class PDESOLVERSGPU_EXPORT FlexibleDiffusionSolverFE_GPU_Device {
    protected:
        LatticeType latticeType;
    public:
        virtual ~FlexibleDiffusionSolverFE_GPU_Device() {}

        FlexibleDiffusionSolverFE_GPU_Device() {}

    protected:
        //allocate resources, prepare device
        virtual void alloc(size_t fieldLen) = 0;

    public:
        //select the GPU device
        virtual void init(int gpuDeviceIndex, LatticeType lt, size_t fieldLen) = 0;


        //filling up solver params and sending them to the GPU
        virtual void prepareSolverParams(Dim3D fieldDim, DiffusionData const &diffData) = 0;

        virtual std::string solverName() = 0;

        virtual void fieldHostToDevice(float const *h_field) = 0;

        virtual void fieldDeviceToHost(float *h_field) const = 0;

        virtual void diffuseSingleField() = 0;

        virtual void swapScratchAndField() = 0;

        virtual void initCellTypeArray(unsigned char *arr, size_t arrLength) = 0;

        virtual void initBoundaryArray(unsigned char *arr, size_t arrLength) = 0;
    };

}//namespace CompuCell3D

#endif