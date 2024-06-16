#include "DiffusionSolverFE_OpenCL.h"

//Ivan Komarov
#include <stdlib.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/steppables/BoxWatcher/BoxWatcher.h>
#include <PublicUtilities/ParallelUtilsOpenMP.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include "../GPUSolverParams.h"
#include "OpenCLHelper.h"
#include <XMLUtils/CC3DXMLElement.h>
#include <algorithm>
#include "OCLNeighbourIndsInfo.h"
#include <Logger/CC3DLogger.h>

#if defined(_WIN32)
#undef max
#undef min
#endif


using namespace CompuCell3D;

DiffusionSolverFE_OpenCL::DiffusionSolverFE_OpenCL(void) :
        DiffusableVectorCommon<float, Array3DCUDA>(),
        oclHelper(NULL),
        d_cellTypes(NULL),
        d_bcIndicator(NULL),
        d_cellIdsAllocated(false),

        gpuDeviceIndex(-1) {
//	QueryPerformanceFrequency(&fq);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::handleEventLocal(CC3DEvent &_event) {
    if (_event.id == LATTICE_RESIZE) {
        // CODE WHICH HANDLES CELL LATTICE RESIZE
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DiffusionSolverFE_OpenCL::~DiffusionSolverFE_OpenCL(void) {
    if (oclHelper) {
        cl_int res;
        oclHelper->Finish();

        res = clReleaseMemObject(d_nbhdConcShifts);
        ASSERT_OR_THROW("Can not release d_nbhdConcShifts", res == CL_SUCCESS);

        res = clReleaseMemObject(d_nbhdDiffShifts);
        ASSERT_OR_THROW("Can not release d_nbhdDiffShifts", res == CL_SUCCESS);

        res = clReleaseMemObject(d_cellTypes);
        ASSERT_OR_THROW("Can not release d_cellTypes", res == CL_SUCCESS);

        res = clReleaseMemObject(d_bcIndicator);
        ASSERT_OR_THROW("Can not release d_bcIndicator", res == CL_SUCCESS);


        if (d_cellIdsAllocated) {
            res = clReleaseMemObject(d_cellIds);
            ASSERT_OR_THROW("Can not release d_cellIds", res == CL_SUCCESS);
        }

        res = clReleaseMemObject(d_solverParams);
        ASSERT_OR_THROW("Can not release d_solverParams", res == CL_SUCCESS);

        res = clReleaseMemObject(d_bcSpecifier);
        ASSERT_OR_THROW("Can not release d_bcSpecifier", res == CL_SUCCESS);

        res = clReleaseMemObject(d_scratchField);
        ASSERT_OR_THROW("Can not release d_scratchField", res == CL_SUCCESS);

        res = clReleaseMemObject(d_concentrationField);
        ASSERT_OR_THROW("Can not release d_concentrationField", res == CL_SUCCESS);

        res = clReleaseKernel(kernelUniDiff);
        ASSERT_OR_THROW("Can not release kernelUniDiff", res == CL_SUCCESS);

        res = clReleaseKernel(kernelBoundaryConditionInit);
        ASSERT_OR_THROW("Can not release kernelBoundaryConditionInit", res == CL_SUCCESS);

        res = clReleaseKernel(kernelBoundaryConditionInitLatticeCorners);
        ASSERT_OR_THROW("Can not release kernelBoundaryConditionInit", res == CL_SUCCESS);

        res = clReleaseKernel(secreteSingleFieldKernel);
        ASSERT_OR_THROW("Can not release secreteSingleFieldKernel", res == CL_SUCCESS);

        res = clReleaseKernel(secreteConstantConcentrationSingleFieldKernel);
        ASSERT_OR_THROW("Can not release secreteConstantConcentrationSingleFieldKernel", res == CL_SUCCESS);

        res = clReleaseKernel(secreteOnContactSingleFieldKernel);
        ASSERT_OR_THROW("Can not release secreteOnContactSingleFieldKernel", res == CL_SUCCESS);

        res = clReleaseProgram(program);
        ASSERT_OR_THROW("Can not release program", res == CL_SUCCESS);

        delete oclHelper;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_OpenCL::secreteSingleField(unsigned int idx) {
    CC3D_Log(LOG_TRACE) << "CALLING SECRETE SINGLE FIELD KERNEL";
    //notice concDevPtr is set elsewhere - no need to set it here
    cl_int errArg = clSetKernelArg(secreteSingleFieldKernel, 0, sizeof(cl_mem), concDevPtr);
    ASSERT_OR_THROW("Can not set secreteSingleFieldKernel  arguments\n", errArg == CL_SUCCESS);

    cl_int err = oclHelper->EnqueueNDRangeKernel(secreteSingleFieldKernel, 3, globalWorkSize, localWorkSize);
    ASSERT_OR_THROW("secreteSingleFieldKernel failed", err == CL_SUCCESS);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::secreteOnContactSingleField(unsigned int idx) {

    //first we initialize h_cellid_field using BC specification
    prepSecreteOnContactSingleField(idx);

    //notice concDevPtr is set elsewhere - no need to set it here
    cl_int errArg = clSetKernelArg(secreteOnContactSingleFieldKernel, 0, sizeof(cl_mem), concDevPtr);
    ASSERT_OR_THROW("Can not set secreteOnContactSingleFieldKernel  arguments\n", errArg == CL_SUCCESS);

    cl_int err = oclHelper->EnqueueNDRangeKernel(secreteOnContactSingleFieldKernel, 3, globalWorkSize, localWorkSize);
    ASSERT_OR_THROW("secreteOnContactSingleFieldKernel failed", err == CL_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Dim3D DiffusionSolverFE_OpenCL::getInternalDim() {
    //notice that because we are using soe of the prewritten fcns from CPU diffusion solvers and CPU diffusion solver workFieldDim is (+1,+1,+1) greater than Cuda (due to extra scratch field built in in CPU field) we increase internalDim here
    //and leave CPU code as ias. This is a hack but for now it will do
    return getConcentrationField(0)->getInternalDim() + Dim3D(1, 1, 1);;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::prepCellId(unsigned int idx) {

    bool detailedBCFlag = bcSpecFlagVec[idx];
    BoundaryConditionSpecifier &bcSpec = bcSpecVec[idx];

    bool periodicX = false, periodicY = false, periodicZ = false;
    float NON_CELL = -2.0; //we assume medium cell id is -1 not zero because normally cells in older versions of CC3D we allwoed cells with id 0 . For that reason we set NON_CEll to -2.0

    Dim3D workFieldDimInternal = getInternalDim();

    if (detailedBCFlag) {
        if (bcSpec.planePositions[MIN_X] == PERIODIC || bcSpec.planePositions[MAX_X] == PERIODIC) {
            periodicX = true;
        }
    } else if (periodicBoundaryCheckVector[0]) {
        periodicX = true;
    }

    if (detailedBCFlag) {
        if (bcSpec.planePositions[MIN_Y] == PERIODIC || bcSpec.planePositions[MAX_Y] == PERIODIC) {
            periodicY = true;
        }
    } else if (periodicBoundaryCheckVector[1]) {
        periodicY = true;
    }

    if (detailedBCFlag) {
        if (bcSpec.planePositions[MIN_Z] == PERIODIC || bcSpec.planePositions[MAX_Z] == PERIODIC) {
            periodicZ = true;
        }
    } else if (periodicBoundaryCheckVector[2]) {
        periodicZ = true;
    }

    if (periodicX) {
        int x = 0;
        for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(fieldDim.x, y, z));
            }

        x = fieldDim.x + 1;
        for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(1, y, z));
            }
    } else {
        int x = 0;
        for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }

        x = fieldDim.x + 1;
        for (int y = 0; y < workFieldDimInternal.y - 1; ++y)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }
    }

    if (periodicY) {
        int y = 0;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(x, fieldDim.y, z));
            }

        y = fieldDim.y + 1;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(x, 1, z));
            }
    } else {
        int y = 0;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }

        y = fieldDim.y + 1;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int z = 0; z < workFieldDimInternal.z - 1; ++z) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }

    }

    if (periodicZ) {
        int z = 0;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(x, y, fieldDim.z));
            }

        z = fieldDim.z + 1;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                h_cellid_field->setDirect(x, y, z, h_cellid_field->getDirect(x, y, 1));
            }

    } else {
        int z = 0;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }

        z = fieldDim.z + 1;
        for (int x = 0; x < workFieldDimInternal.x - 1; ++x)
            for (int y = 0; y < workFieldDimInternal.y - 1; ++y) {
                h_cellid_field->setDirect(x, y, z, NON_CELL);
            }
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::prepSecreteOnContactSingleField(unsigned int idx) {

    if (!iterationNumber) { //only first call to secreteOnContact for a given MCS will prep cellId field
        prepCellId(idx);
    }

    if (!d_cellIdsAllocated) {

        size_t mem_size_cellid_field = h_cellid_field->getArraySize() * sizeof(float);
        d_cellIds = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_cellid_field);

        cl_int err = clSetKernelArg(secreteOnContactSingleFieldKernel, 0, sizeof(cl_mem), &d_concentrationField);
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 1, sizeof(cl_mem), &d_cellTypes);
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 2, sizeof(cl_mem), &d_cellIds);
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 3, sizeof(cl_mem), &d_solverParams);
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 4, sizeof(cl_mem), &d_nbhdConcShifts);
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 5,
                                   sizeof(unsigned char) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) *
                                   (localWorkSize[2] + 2), NULL);//local cell type
        err = err | clSetKernelArg(secreteOnContactSingleFieldKernel, 6,
                                   sizeof(float) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) *
                                   (localWorkSize[2] + 2), NULL);//local cell type
        ASSERT_OR_THROW("Can not set secreteOnContactSingleFieldKernel  arguments\n", err == CL_SUCCESS);


        d_cellIdsAllocated = true;

    }

    if (!iterationNumber) { //only first call to secrete on contact needs to transfer cellid field to device
        ASSERT_OR_THROW("Can not write transfer h_cellid_field to the device\n",
                        oclHelper->WriteBuffer(d_cellIds, h_cellid_field->getContainer(),
                                               h_cellid_field->getArraySize()) == CL_SUCCESS);

        ASSERT_OR_THROW("Can not set  secreteOnContactSingleFieldKernel d_cellIds argument\n",
                        clSetKernelArg(secreteOnContactSingleFieldKernel, 2, sizeof(cl_mem), &d_cellIds) == CL_SUCCESS);
    }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::secreteConstantConcentrationSingleField(unsigned int idx) {


    //notice concDevPtr is set elsewhere - no need to set it here
    cl_int errArg = clSetKernelArg(secreteConstantConcentrationSingleFieldKernel, 0, sizeof(cl_mem), concDevPtr);
    ASSERT_OR_THROW("Can not set secreteConstantConcentrationSingleFieldKernel  arguments\n", errArg == CL_SUCCESS);

    cl_int err = oclHelper->EnqueueNDRangeKernel(secreteConstantConcentrationSingleFieldKernel, 3, globalWorkSize,
                                                 localWorkSize);
    ASSERT_OR_THROW("secreteConstantConcentrationSingleFieldKernel failed", err == CL_SUCCESS);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//initializes and transfers to GPU BCSpecifier structure - nothing else. Actual BC initialization is done inside boundaryConditionInitKernel on GPU
void DiffusionSolverFE_OpenCL::boundaryConditionGPUSetup(int idx) {
    // in the GPU Solver we initialize boundary conditions in the kernel
    CC3D_Log(LOG_TRACE) << "INITIALIZING BC IN THE GPU CODE OVERLOADED FCN";

    //***************** BC INIT
    bool detailedBCFlag = bcSpecFlagVec[idx];
    BoundaryConditionSpecifier &bcSpec = bcSpecVec[idx];

    BCSpecifier bcSpecifier;
    //by default we set all BC's to periodic - note due to OpenCL limitation we cannot include constructory for the BCSpecifier structure - hence manunal initialization here
    for (int i = 0; i < 6; ++i) {
        bcSpecifier.planePositions[i] = PERIODIC;
        bcSpecifier.values[i] = 0.0;
    }
    CC3D_Log(LOG_TRACE) << "detailedBCFlag="<<detailedBCFlag;
    if (detailedBCFlag) {
        for (int i = 0; i < 6; ++i) {
            bcSpecifier.planePositions[i] = bcSpec.planePositions[i];
            bcSpecifier.values[i] = bcSpec.values[i];
        }
    } else {
        if (periodicBoundaryCheckVector[0]) {//periodic boundary conditions were set in x direction
            bcSpecifier.planePositions[MIN_X] = PERIODIC;
            bcSpecifier.planePositions[MAX_X] = PERIODIC;
        } else {//noFlux BC
            bcSpecifier.planePositions[MIN_X] = CONSTANT_DERIVATIVE;
            bcSpecifier.planePositions[MAX_X] = CONSTANT_DERIVATIVE;
            bcSpecifier.values[MIN_X] = 0.0;
            bcSpecifier.values[MAX_X] = 0.0;

        }

        if (periodicBoundaryCheckVector[1]) {//periodic boundary conditions were set in y direction
            bcSpecifier.planePositions[MIN_Y] = PERIODIC;
            bcSpecifier.planePositions[MAX_Y] = PERIODIC;

        } else {//noFlux BC
            bcSpecifier.planePositions[MIN_Y] = CONSTANT_DERIVATIVE;
            bcSpecifier.planePositions[MAX_Y] = CONSTANT_DERIVATIVE;
            bcSpecifier.values[MIN_Y] = 0.0;
            bcSpecifier.values[MAX_Y] = 0.0;
        }

        if (periodicBoundaryCheckVector[2]) {//periodic boundary conditions were set in z direction
            bcSpecifier.planePositions[MIN_Z] = PERIODIC;
            bcSpecifier.planePositions[MAX_Z] = PERIODIC;

        } else {//noFlux BC
            bcSpecifier.planePositions[MIN_Z] = CONSTANT_DERIVATIVE;
            bcSpecifier.planePositions[MAX_Z] = CONSTANT_DERIVATIVE;
            bcSpecifier.values[MIN_Z] = 0.0;
            bcSpecifier.values[MAX_Z] = 0.0;
        }

    }


    // for (int i = 0 ; i < 6 ; ++i){
//    CC3D_Log(LOG_TRACE) << "planePositions["<<i<<"]="<<bcSpecifier.planePositions[i];
//    CC3D_Log(LOG_TRACE) << "values["<<i<<"]="<<bcSpecifier.values[i];
    // }        

    cl_int err = oclHelper->WriteBuffer(d_bcSpecifier, &bcSpecifier, 1);
    ASSERT_OR_THROW("Can not copy Cell Type field to GPU", err == CL_SUCCESS);


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void DiffusionSolverFE_OpenCL::boundaryConditionInit(int idx) {


}

void DiffusionSolverFE_OpenCL::stepImpl(const unsigned int _currentStep) {


    for (unsigned int i = 0; i < diffSecrFieldTuppleVec.size(); ++i) {
        this->boundaryConditionGPUSetup(
                i); //initializes and transfers to GPU BCSpecifier structure - nothing else. Actual BC initialization is done inside boundaryConditionInitKernel on GPU. Notice: secreteOnContactKernel also uses bcSpecifier



        //initializing ptrs to conc and scratch fields
        concDevPtr = &d_concentrationField;
        scratchDevPtr = &d_scratchField;

        DiffusionData &diffData = diffSecrFieldTuppleVec[i].diffData;
        SecretionData &secrData = diffSecrFieldTuppleVec[i].secrData;

        ConcentrationField_t &concentrationField = *(this)->getConcentrationField(i);

        ASSERT_OR_THROW("Coupling Terms are not supported yet", !haveCouplingTerms);
        ASSERT_OR_THROW("Box watcher is not supported yet", !diffData.useBoxWatcher);
        ASSERT_OR_THROW("Threshold is not supported yet", !diffData.useThresholds);

        prepCellTypeField(i); // here we initialize celltype array  boundaries - we do it once per  MCS

        initCellTypesAndBoundariesImpl(); //this sends type array to device - most likely this can be done one per stepImpl call. For now leaving it here
        ///////////

        // boundaryConditionInit(i);
        CC3D_Log(LOG_TRACE) << "diffuseSingleFieldImpl: THIS IS EXTRA TIMES PER MCS="<<diffData.extraTimesPerMCS;

        SetSolverParams(diffData,
                        secrData); //transfer diffusion data and other useful parameters describing PDE to device

        float *h_Field = concentrationField.getContainer();

        fieldHostToDevice(h_Field);//transfer conc field to device

        iterationNumber = 0; // this variable is important because other routines can sense if this is first or subsequent call to diffuse or secrete functions. Some work in this functions has to be done during initial call and skipped in others

        if (scaleSecretion) {
            CC3D_Log(LOG_TRACE) << "DIFFUSION SOLVER REGULAR NON_FLEXIBLE";
            if (!scalingExtraMCSVec[i]) { //we do not call diffusion step but call secretion - this happens when diffusion const is 0 but we still want to have secretion
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
                }
            }

            for (int extraMCS = 0; extraMCS < scalingExtraMCSVec[i]; extraMCS++) {
                diffuseSingleField(i); // this also initializes boundary conditions
                for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                    (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
                }
                iterationNumber++;
            }
        } else { //solver behaves as FlexibleDiffusionSolver - i.e. secretion is done at once followed by multiple diffusive steps
            for (unsigned int j = 0; j < diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec.size(); ++j) {
                (this->*diffSecrFieldTuppleVec[i].secrData.secretionFcnPtrVec[j])(i);
            }

            for (int extraMCS = 0; extraMCS < scalingExtraMCSVec[i]; extraMCS++) {
                diffuseSingleField(i); // this also initializes boundary conditions
                iterationNumber++;
            }


        }


        oclHelper->Finish();
        fieldDeviceToHost(h_Field); //transfer conc field to back to host memory

    }

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_OpenCL::diffuseSingleField(unsigned int idx) {

    cl_mem *tmpDevPtr;
    cl_int errArgBCI;
    cl_int errBCI;


    // //initialize boundary conditions
    errArgBCI = clSetKernelArg(kernelBoundaryConditionInit, 0, sizeof(cl_mem), concDevPtr);
    ASSERT_OR_THROW("Can not set boundaryConditionInitKernel  arguments\n", errArgBCI == CL_SUCCESS);
    CC3D_Log(LOG_TRACE) << "globalWorkSize=["<<globalWorkSize[0]<<","<<globalWorkSize[1]<<","<<globalWorkSize[2]<<"']";
    CC3D_Log(LOG_TRACE) << "localWorkSize=["<<localWorkSize[0]<<","<<localWorkSize[1]<<","<<localWorkSize[2]<<"']";

    errBCI = oclHelper->EnqueueNDRangeKernel(kernelBoundaryConditionInit, 3, globalWorkSize, localWorkSize);
    ASSERT_OR_THROW("kernelBoundaryConditionInit failed", errBCI == CL_SUCCESS);

    if (latticeType != SQUARE_LATTICE) {
        //initialize boundary conditions Lattice Corners - necessary only for hex lattice
        errArgBCI = clSetKernelArg(kernelBoundaryConditionInitLatticeCorners, 0, sizeof(cl_mem), concDevPtr);
        ASSERT_OR_THROW("Can not set boundaryConditionInitLatticeCornersKernel  arguments\n", errArgBCI == CL_SUCCESS);

        errBCI = oclHelper->EnqueueNDRangeKernel(kernelBoundaryConditionInitLatticeCorners, 3, globalWorkSize,
                                                 localWorkSize);
        ASSERT_OR_THROW("kernelBoundaryConditionInitLatticeCorners failed", errBCI == CL_SUCCESS);
    }

    //diffusion step
    cl_int errArg = clSetKernelArg(kernelUniDiff, concFieldArgPosition, sizeof(cl_mem), concDevPtr);
    errArg = errArg | clSetKernelArg(kernelUniDiff, scratchFieldArgPosition, sizeof(cl_mem), scratchDevPtr);

    errArg = errArg | clSetKernelArg(kernelUniDiff, bcSpecifierArgPosition, sizeof(cl_mem), &d_bcSpecifier);
    errArg = errArg | clSetKernelArg(kernelUniDiff, bcIndicatorArgPosition, sizeof(cl_mem), &d_bcIndicator);

    ASSERT_OR_THROW("Setting concentration and scratch field arguments for diffusion kernel failed",
                    errArg == CL_SUCCESS);


    cl_int err = oclHelper->EnqueueNDRangeKernel(kernelUniDiff, 3, globalWorkSize, localWorkSize);
    if (err != CL_SUCCESS)
        CC3D_Log(LOG_DEBUG) << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Diffusion Kernel failed", err == CL_SUCCESS);


    //swapping pointers    
    tmpDevPtr = concDevPtr;
    concDevPtr = scratchDevPtr;
    scratchDevPtr = tmpDevPtr;


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::SetConstKernelArguments() {
    int kArg = 0;
    cl_int err;

    /// UniDiff
    kArg = 0;

    concFieldArgPosition = kArg++;
    err = clSetKernelArg(kernelUniDiff, concFieldArgPosition, sizeof(cl_mem), &d_concentrationField);

    scratchFieldArgPosition = kArg++;
    err = err | clSetKernelArg(kernelUniDiff, scratchFieldArgPosition, sizeof(cl_mem), &d_scratchField);

    err = err | clSetKernelArg(kernelUniDiff, kArg++, sizeof(cl_mem), &d_solverParams);

    bcSpecifierArgPosition = kArg++;
    err = err | clSetKernelArg(kernelUniDiff, bcSpecifierArgPosition, sizeof(cl_mem), &d_bcSpecifier);
    err = err | clSetKernelArg(kernelUniDiff, kArg++, sizeof(cl_mem), &d_cellTypes);

    bcIndicatorArgPosition = kArg++;
    err = err | clSetKernelArg(kernelUniDiff, bcIndicatorArgPosition, sizeof(cl_mem), &d_bcIndicator);

    err = err | clSetKernelArg(kernelUniDiff, kArg++, sizeof(cl_mem), &d_nbhdConcShifts);
    err = err | clSetKernelArg(kernelUniDiff, kArg++, sizeof(cl_mem), &d_nbhdDiffShifts);

    err = err | clSetKernelArg(kernelUniDiff, kArg++,
                               sizeof(float) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) * (localWorkSize[2] + 2),
                               NULL);//local field
    err = err | clSetKernelArg(kernelUniDiff, kArg++,
                               sizeof(unsigned char) * (localWorkSize[0] + 2) * (localWorkSize[1] + 2) *
                               (localWorkSize[2] + 2), NULL);//local cell type

    ASSERT_OR_THROW("Can not set uniDiff kernel's arguments\n", err == CL_SUCCESS);


    ///********************************** BC Kernel
    err = clSetKernelArg(kernelBoundaryConditionInit, 0, sizeof(cl_mem), &d_concentrationField);
    err = err | clSetKernelArg(kernelBoundaryConditionInit, 1, sizeof(cl_mem), &d_solverParams);
    err = err | clSetKernelArg(kernelBoundaryConditionInit, 2, sizeof(cl_mem), &d_bcSpecifier);
    ASSERT_OR_THROW("Can not set boundaryConditionInitKernel  arguments\n", err == CL_SUCCESS);


    ///********************************** BC Kernel Lattice Corners
    err = clSetKernelArg(kernelBoundaryConditionInitLatticeCorners, 0, sizeof(cl_mem), &d_concentrationField);
    err = err | clSetKernelArg(kernelBoundaryConditionInitLatticeCorners, 1, sizeof(cl_mem), &d_solverParams);
    err = err | clSetKernelArg(kernelBoundaryConditionInitLatticeCorners, 2, sizeof(cl_mem), &d_bcSpecifier);
    ASSERT_OR_THROW("Can not set boundaryConditionInitLatticeCornersKernel  arguments\n", err == CL_SUCCESS);


    ///********************************** secreteSingleFieldKernel
    err = clSetKernelArg(secreteSingleFieldKernel, 0, sizeof(cl_mem), &d_concentrationField);
    err = err | clSetKernelArg(secreteSingleFieldKernel, 1, sizeof(cl_mem), &d_cellTypes);
    err = err | clSetKernelArg(secreteSingleFieldKernel, 2, sizeof(cl_mem), &d_solverParams);
    ASSERT_OR_THROW("Can not set secreteSingleFieldKernel  arguments\n", err == CL_SUCCESS);

    ///********************************** secreteConstantConcentrationSingleFieldKernel
    err = clSetKernelArg(secreteConstantConcentrationSingleFieldKernel, 0, sizeof(cl_mem), &d_concentrationField);
    err = err | clSetKernelArg(secreteConstantConcentrationSingleFieldKernel, 1, sizeof(cl_mem), &d_cellTypes);
    err = err | clSetKernelArg(secreteConstantConcentrationSingleFieldKernel, 2, sizeof(cl_mem), &d_solverParams);
    ASSERT_OR_THROW("Can not set secreteConstantConcentrationSingleFieldKernel  arguments\n", err == CL_SUCCESS);


    ///********************************** secreteConstantConcentrationSingleFieldKernel    
    // arguments for this call are set in prepSecreteOnContactSingleField fcn

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::SetSolverParams(DiffusionData &diffData, SecretionData &secrData) {

    UniSolverParams_t h_solverParams;
    for (int i = 0; i < UCHAR_MAX + 1; ++i) {

        h_solverParams.diffCoef[i] = diffData.diffCoef[i];
        h_solverParams.decayCoef[i] = diffData.decayCoef[i];

        h_solverParams.secretionData[i][SECRETION_CONST] = h_solverParams.secretionData[i][MAX_UPTAKE] = h_solverParams.secretionData[i][RELATIVE_UPTAKE] = 0.0; //zeroing all secreData
        h_solverParams.secretionConstantConcentrationData[i] = 0.0;
        for (int j = 0; j < UCHAR_MAX + 2; ++j) {
            h_solverParams.secretionOnContactData[i][j] = 0.0;
        }

    }

    //assigning secretion data 

    for (std::map<unsigned char, float>::iterator mitr = secrData.typeIdSecrConstMap.begin();
         mitr != secrData.typeIdSecrConstMap.end(); ++mitr) {
        h_solverParams.secretionData[mitr->first][SECRETION_CONST] = mitr->second;
    }

    for (std::map<unsigned char, UptakeData>::iterator mitr = secrData.typeIdUptakeDataMap.begin();
         mitr != secrData.typeIdUptakeDataMap.end(); ++mitr) {
        h_solverParams.secretionData[mitr->first][MAX_UPTAKE] = mitr->second.maxUptake;
        h_solverParams.secretionData[mitr->first][RELATIVE_UPTAKE] = mitr->second.relativeUptakeRate;
    }

    if (secrData.typeIdUptakeDataMap.size()) {
        h_solverParams.secretionDoUptake = 1;
    } else {
        h_solverParams.secretionDoUptake = 0;
    }


    //assigning secretionConstantConcentrationData
    for (std::map<unsigned char, float>::iterator mitr = secrData.typeIdSecrConstConstantConcentrationMap.begin();
         mitr != secrData.typeIdSecrConstConstantConcentrationMap.end(); ++mitr) {
        h_solverParams.secretionConstantConcentrationData[mitr->first] = mitr->second;
    }

    //assigning secretionOnContactData
    for (std::map<unsigned char, SecretionOnContactData>::iterator mitr = secrData.typeIdSecrOnContactDataMap.begin();
         mitr != secrData.typeIdSecrOnContactDataMap.end(); ++mitr) {
        unsigned char cell_type = mitr->first;
        SecretionOnContactData &secretionOnContactData = mitr->second;
        // std::map<unsigned char, float> & contactCellMap=mitr->second;


        for (std::map<unsigned char, float>::iterator mitr_ccm = secretionOnContactData.contactCellMap.begin();
             mitr_ccm != secretionOnContactData.contactCellMap.end(); ++mitr_ccm) {
            unsigned char contact_cell_type = mitr_ccm->first;

            h_solverParams.secretionOnContactData[cell_type][contact_cell_type] = mitr_ccm->second;
            h_solverParams.secretionOnContactData[cell_type][UCHAR_MAX +
                                                             1] = 1.0; // setting flag that indicates that cell_type has secrete on contact data set on

        }
    }


//	h_solverParams.dt=diffData.deltaT;

    h_solverParams.extraTimesPerMCS=diffData.extraTimesPerMCS;    
	h_solverParams.dx=diffData.deltaX;
    h_solverParams.dt=diffData.deltaT;
	h_solverParams.hexLattice=(latticeType==HEXAGONAL_LATTICE);
	h_solverParams.nbhdConcLen=nbhdConcLen;
	h_solverParams.nbhdDiffLen=nbhdDiffLen;
    CC3D_Log(LOG_DEBUG) << "h_solverParams.nbhdConcLen="<<h_solverParams.nbhdConcLen<<" h_solverParams.nbhdDiffLen="<<h_solverParams.nbhdDiffLen;


    h_solverParams.xDim = fieldDim.x;
    h_solverParams.yDim = fieldDim.y;
    h_solverParams.zDim = fieldDim.z;


    oclHelper->WriteBuffer(d_solverParams, &h_solverParams, 1);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::solverSpecific(CC3DXMLElement *_xmlData){
	//getting requested GPU device index
	if(_xmlData->findElement("GPUDeviceIndex")){
		gpuDeviceIndex=_xmlData->getFirstElement("GPUDeviceIndex")->getInt();
        CC3D_Log(LOG_DEBUG) << "GPU device #"<<gpuDeviceIndex<<" requested";
	}else{
        CC3D_Log(LOG_DEBUG) << "No specific GPU requested, it will be selected automatically";
		gpuDeviceIndex=-1;
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::initImpl() {

    if (gpuDeviceIndex == -1)

        gpuDeviceIndex = 0;
    CC3D_Log(LOG_TRACE) << "Requested GPU device index is "<<gpuDeviceIndex;

    oclHelper = new OpenCLHelper(gpuDeviceIndex);


    localWorkSize[0] = BLOCK_SIZE;

    localWorkSize[1] = BLOCK_SIZE;

    // TODO: BLOCK size can be non-optimal in terms of maximum performance
    // on some low-end GPUs we need to limit  number of workers because if the kernel code is too large those cards will go out of  resources
    // here we put heuristic 0.8 factor limiting number of threads - this will give slightly worse performence but will avoid crashes

    localWorkSize[2] = std::min(oclHelper->getMaxWorkGroupSize() / (BLOCK_SIZE * BLOCK_SIZE), size_t(fieldDim.z));

    // This was crashing on an AMD Radeon.  But the Tesla V100 is high end.  So remove factor.
    //size_t optimal_z_dim=0.8*(oclHelper->getMaxWorkGroupSize()/(BLOCK_SIZE*BLOCK_SIZE));
    //localWorkSize[2]=std::min( optimal_z_dim ,  size_t(fieldDim.z));



    int xReminder = fieldDim.x % localWorkSize[0];

    int yReminder = fieldDim.y % localWorkSize[1];

    int zReminder = fieldDim.z % localWorkSize[2];


    // we add extra layer of localWorkSize[0] in case fieldDim.x is not divisible by localWorkSize[0]
    globalWorkSize[0] = fieldDim.x + ((fieldDim.x % localWorkSize[0]) ? localWorkSize[0] : 0) - xReminder;

    // we add extra layer of localWorkSize[1] in case fieldDim.y is not divisible by localWorkSize[1]
    globalWorkSize[1] = fieldDim.y + ((fieldDim.y % localWorkSize[1]) ? localWorkSize[1] : 0) - yReminder;

    // we add extra layer of localWorkSize[2] in case fieldDim.z is not divisible by localWorkSize[2]
    globalWorkSize[2] = fieldDim.z + ((fieldDim.z % localWorkSize[2]) ? localWorkSize[2] : 0) - zReminder;


    // we add extra layer of localWorkSize[0] in case fieldDim.x is not divisible by localWorkSize[0]
    // globalWorkSize[0]=fieldDim.x+( fieldDim.x % fieldDim.x ? localWorkSize[0] : 0); 

    // we add extra layer of localWorkSize[1] in case fieldDim.y is not divisible by localWorkSize[1]
    // globalWorkSize[1]=fieldDim.y+( fieldDim.y % fieldDim.y ? localWorkSize[1] : 0); 

    // globalWorkSize[2]=fieldDim.z;


    field_len = h_celltype_field->getArraySize();

    gpuAlloc(field_len);
    CC3D_Log(LOG_DEBUG) << "building OpenCL program";

    //const char *kernelSource[] = { "lib/CompuCell3DSteppables/OpenCL/GPUSolverParams.h",

    //    "lib/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl" };

//    const char *kernelSource[] = {
//            (string("lib/site-packages/cc3d/cpp/CompuCell3DSteppables/OpenCL/GPUSolverParams.h")).c_str(),
//            (string("lib/site-packages/cc3d/cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl")).c_str()};
//

    char *cc3d_opencl_solvers_dir = getenv("CC3D_OPENCL_SOLVERS_DIR");
    ASSERT_OR_THROW("CC3D_OPENCL_SOLVERS_DIR environment variable is not set. Cannot run DiffusionSolverFE_OpenCL without it. Please set this environment variable to so that it points to the directory containing GPUSolverParams.h and DiffusionKernel.cl", cc3d_opencl_solvers_dir)
    string cc3d_open_cl_solvers_dir = string(cc3d_opencl_solvers_dir);



//    const char *kernelSource[] = {
//            (string("c:/miniconda3/envs/cc3d_460_310_develop/Lib/site-packages/cc3d/cpp/CompuCell3DSteppables/OpenCL/GPUSolverParams.h")).c_str(),
//            (string("c:/miniconda3/envs/cc3d_460_310_develop/Lib/site-packages/cc3d/cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl")).c_str()
//            };

    string solver_params = cc3d_open_cl_solvers_dir+string("/GPUSolverParams.h");
    string diffusion_kernel = cc3d_open_cl_solvers_dir+string("/DiffusionKernel.cl");
    const char *kernelSource[] = {
            solver_params.c_str(),
            diffusion_kernel.c_str()
            };


    if (!oclHelper->LoadProgram(kernelSource, 2, program)) {

        ASSERT_OR_THROW("Can't load the OpenCL kernel", false);

    }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_OpenCL::gpuAlloc(size_t fieldLen){
    CC3D_Log(LOG_DEBUG) << "Allocating GPU memory for the field of length "<<fieldLen;
    CC3D_Log(LOG_DEBUG) << "Field dimensions are: "<<fieldDim.x<<" "<<fieldDim.y<<" "<<fieldDim.z;

    size_t mem_size_field = fieldLen * sizeof(float);
    size_t mem_size_celltype_field = fieldLen * sizeof(unsigned char);
    size_t mem_size_bcIndicator_field = fieldLen * sizeof(signed char);

    // // // d_concentrationField=oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_field);
    d_concentrationField = oclHelper->CreateBuffer(CL_MEM_READ_WRITE, mem_size_field);
    d_cellTypes = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_celltype_field);
    d_bcIndicator = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, mem_size_bcIndicator_field);
    d_scratchField = oclHelper->CreateBuffer(CL_MEM_WRITE_ONLY, mem_size_field);

    d_solverParams = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(UniSolverParams_t));

    d_bcSpecifier = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(BCSpecifier));

    //allocating secretionData - 2D array indexed by type entries are {secretion constant, maxUptake, relativeUptake}
    size_t secrDataSize = 3 * (UCHAR_MAX + 1) * sizeof(float);
    d_secretionData = oclHelper->CreateBuffer(CL_MEM_READ_WRITE, secrDataSize);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_OpenCL::extraInitImpl() {

    CreateKernel();
    int layers;

    try {
        OCLNeighbourIndsInfo onii = OCLNeighbourIndsInfo::Init(latticeType, fieldDim, getBoundaryStrategy(),
                                                               hexOffsetArray, offsetVecCartesian);


        //IMPORTANT: these two variables are crucial and they have to be set here otherwise opencl kernel will not work properly . They determine the size of offset vectors
        // nbhdConcLen=onii.mh_nbhdConcShifts.size();
        // nbhdDiffLen=onii.mh_nbhdDiffShifts.size();
        nbhdConcLen = onii.m_nbhdConcLen;
        nbhdDiffLen = onii.m_nbhdDiffLen;

        d_nbhdDiffShifts = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4) * onii.mh_nbhdDiffShifts.size());
        d_nbhdConcShifts = oclHelper->CreateBuffer(CL_MEM_READ_ONLY, sizeof(cl_int4) * onii.mh_nbhdConcShifts.size());

        cl_int err = oclHelper->WriteBuffer(d_nbhdDiffShifts, &onii.mh_nbhdDiffShifts[0],
                                            onii.mh_nbhdDiffShifts.size());

		err = err | oclHelper->WriteBuffer(d_nbhdConcShifts, &onii.mh_nbhdConcShifts[0], onii.mh_nbhdConcShifts.size());
		ASSERT_OR_THROW("Can not initialize shifts", err==CL_SUCCESS);

        //set the arguements of our kernel
        SetConstKernelArguments();
    } catch (...) {
        ASSERT_OR_THROW("exception caught", false);
    }

    CC3D_Log(LOG_DEBUG) << "extraInitImpl finished";
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::fieldHostToDevice(float const *h_field){
	//CheckConcentrationField(h_field);

    ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);
//	LARGE_INTEGER tb, te;
//	QueryPerformanceCounter(&tb);
    if (oclHelper->WriteBuffer(d_concentrationField, h_field, field_len) != CL_SUCCESS) {
        ASSERT_OR_THROW("Can not write to device buffer", false);
    }

    //to preserve boundary layers
    if (oclHelper->CopyBuffer<float>(d_concentrationField, d_scratchField, field_len) != CL_SUCCESS) {
        ASSERT_OR_THROW("Can not copy device buffer", false);
    }
//	QueryPerformanceCounter(&te);
//	totalTransferTime+=(double)(te.QuadPart-tb.QuadPart)/(double)fq.QuadPart;

}

void DiffusionSolverFE_OpenCL::fieldDeviceToHost(float *h_field) const {
    ASSERT_OR_THROW("oclHelper object must be initialized", oclHelper);
    CC3D_Log(LOG_TRACE) << "BUFFER LENGTH="<<field_len<<" "<<h_celltype_field->getArraySize();
    if (oclHelper->ReadBuffer(*concDevPtr, h_field, field_len) != CL_SUCCESS) {
        ASSERT_OR_THROW("Can not read from device buffer", false);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void DiffusionSolverFE_OpenCL::CreateKernel() {
    //initialize our kernel from the program

    cl_int err;

    kernelUniDiff = clCreateKernel(program, "uniDiff", &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernelUniDiff uniDiff: " << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create a kernelUniDiff", err == CL_SUCCESS);


    kernelBoundaryConditionInit = clCreateKernel(program, "boundaryConditionInitKernel", &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel boundaryConditionInitKernel: " << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create a boundaryConditionInitKernel", err == CL_SUCCESS);

    kernelBoundaryConditionInitLatticeCorners = clCreateKernel(program, "boundaryConditionInitLatticeCornersKernel",
                                                               &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel boundaryConditionInitLatticeCornersKernel: " << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create a boundaryConditionInitLatticeCornersKernel", err == CL_SUCCESS);


    secreteSingleFieldKernel = clCreateKernel(program, "secreteSingleFieldKernel", &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel secreteSingleFieldKernel: " << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create secreteSingleFieldKernel", err == CL_SUCCESS);


    secreteConstantConcentrationSingleFieldKernel = clCreateKernel(program,
                                                                   "secreteConstantConcentrationSingleFieldKernel",
                                                                   &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel secreteConstantConcentrationSingleFieldKernel: " << oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create secreteConstantConcentrationSingleFieldKernel", err == CL_SUCCESS);


    secreteOnContactSingleFieldKernel = clCreateKernel(program, "secreteOnContactSingleFieldKernel", &err);
    CC3D_Log(LOG_DEBUG) << "clCreateKernel for kernel secreteOnContactSingleFieldKernel: " <<  oclHelper->ErrorString(err);
    ASSERT_OR_THROW("Can not create secreteOnContactSingleFieldKernel", err == CL_SUCCESS);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::initSecretionData() {
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DiffusionSolverFE_OpenCL::initCellTypesAndBoundariesImpl() {
    cl_int err = oclHelper->WriteBuffer(d_cellTypes, h_celltype_field->getContainer(), field_len);
    ASSERT_OR_THROW("Can not copy Cell Type field to GPU", err == CL_SUCCESS);

    err = oclHelper->WriteBuffer(d_bcIndicator, bc_indicator_field->getContainer(), field_len);
    ASSERT_OR_THROW("Can not copy Boundary Condition Indicator fields field to GPU", err == CL_SUCCESS);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void DiffusionSolverFE_OpenCL::finish() {

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string DiffusionSolverFE_OpenCL::toStringImpl() {
    return "DiffusionSolverFE_OpenCL";
}
