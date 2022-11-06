#include "ImplicitMatrix.h"

#include <sstream>

#include "../GPUSolverParams.h"
#include "GPUBoundaryConditions.h"
/*
#ifndef PDESOLVERS_EXPORT
#define PDESOLVERS_EXPORT
#endif

#include <BoundaryConditionSpecifier.h>*/
#include <CompuCell3D/CC3DExceptions.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <Logger/CC3DLogger.h>
using namespace CompuCell3D;
static const char *programPath = "ImplicitMatrix.cl";

ImplicitMatrix::ImplicitMatrix(OpenCLHelper const &oclHelper, UniSolverParams const &solverParams,
                               cl_mem const &d_CellTypes,
                               cl_mem const &d_outputfield, GPUBoundaryConditions const &boundaryConditions,
                               std::string const &pathToKernels) :
        m_oclHelper(oclHelper), mh_solverParams(solverParams), md_cellTypes(d_CellTypes), isTimeStepSet(false),
        md_solverParams(oclHelper, CL_MEM_READ_ONLY, sizeof(UniSolverParams) * 1, &solverParams),
        //md_outputField(oclHelper, CL_MEM_READ_WRITE, sizeof(float)*fieldLength(), NULL),
        md_outputField(d_outputfield),
        md_boundaryConditions(oclHelper, CL_MEM_READ_ONLY, sizeof(GPUBoundaryConditions) * 1, &boundaryConditions),
        mh_boundaryConditions(boundaryConditions) {

	//loading OpenCL program
	std::string fns[]={
		pathToKernels+"GPUSolverParams.h",
		pathToKernels+"GPUBoundaryConditions.h",
		pathToKernels+"common.cl",
		pathToKernels+"ImplicitMatrix.cl"
	};
	const char *programPaths[]={fns[0].c_str(), //TODO: find size of an array automatically
		fns[1].c_str(),
		fns[2].c_str(),
		fns[3].c_str()};
	CC3D_Log(LOG_DEBUG) << "OpenCL kernel names for ImplicitMatrix:";
    for (int i = 0; i < 4; ++i) {
        CC3D_Log(LOG_DEBUG) <<"\t"<<programPaths[i];
    }

    if (!oclHelper.LoadProgram(programPaths, 4, m_clProgram)) {
        throw std::runtime_error("Can't create ImplicitMatrix object, OpenCL program creation failed");
    }

    //creating kernel
    m_prodCoreKernel = new OpenCLKernel(m_clProgram, "ImplicitMatrixProdCore");
    m_prodBoundariesKernel = new OpenCLKernel(m_clProgram, "ImplicitMatrixProdBoundaries");
    m_modifyRHStoBC = new OpenCLKernel(m_clProgram, "ApplyBCToRHS");


    //setting kernel arguments
    m_prodCoreKernel->setArgument(1, md_solverParams.buffer());
    m_prodCoreKernel->setArgument(2, md_cellTypes);
    m_prodCoreKernel->setArgument(4, md_outputField);

    m_prodBoundariesKernel->setArgument(1, md_solverParams.buffer());
    m_prodBoundariesKernel->setArgument(2, md_cellTypes);
    m_prodBoundariesKernel->setArgument(4, md_boundaryConditions.buffer());
    m_prodBoundariesKernel->setArgument(5, md_outputField);

    m_modifyRHStoBC->setArgument(1, md_solverParams.buffer());
    m_modifyRHStoBC->setArgument(2, md_cellTypes);
    m_modifyRHStoBC->setArgument(4, md_boundaryConditions.buffer());

}

void ImplicitMatrix::setdt(float dt) const {
    try {
        m_prodCoreKernel->setArgument(0, dt);
        m_prodBoundariesKernel->setArgument(0, dt);
        m_modifyRHStoBC->setArgument(0, dt);
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    isTimeStepSet = true;
}


ImplicitMatrix::~ImplicitMatrix() {
    clReleaseProgram(m_clProgram);

    delete m_prodCoreKernel;
    delete m_prodBoundariesKernel;
    delete m_modifyRHStoBC;
}

cl_mem ImplicitMatrix::prod(cl_mem xVct) const {
    prodCore(xVct);
    prodBoundaries(xVct);
    return md_outputField;
}

void ImplicitMatrix::prodCore(cl_mem xVct) const {

    ASSERT_OR_THROW("ImplicitMatrix::prodCore: set the time step before calling this function", isTimeStepSet);

    try {
        m_prodCoreKernel->setArgument(3, xVct);
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    //size_t glob_size[]={f};
    size_t glob_size[] = {(size_t) mh_solverParams.xDim, (size_t) mh_solverParams.yDim, (size_t) mh_solverParams.zDim};
    cl_int err = m_oclHelper.EnqueueNDRangeKernel(m_prodCoreKernel->getKernel(), 3, glob_size, NULL);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "Can't compute core part of ImplicitMatrix-vector product: " << m_oclHelper.ErrorString(err);
        ASSERT_OR_THROW(sstr.str().c_str(), false);
    }
}

Dim3D ImplicitMatrix::domainSize() const {
    return Dim3D(mh_solverParams.xDim, mh_solverParams.yDim, mh_solverParams.zDim);
}

void ImplicitMatrix::prodBoundaries(cl_mem xVct) const {
    ASSERT_OR_THROW("ImplicitMatrix::prodBoundaries: set the time step before calling this function", isTimeStepSet);

    try {
        m_prodBoundariesKernel->setArgument(3, xVct);
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    //size_t glob_size[]={f};
    size_t glob_size[] = {(size_t) std::max(mh_solverParams.xDim, mh_solverParams.yDim),
                          (size_t) std::max(mh_solverParams.yDim, mh_solverParams.zDim)};
    cl_int err = m_oclHelper.EnqueueNDRangeKernel(m_prodBoundariesKernel->getKernel(), 2, glob_size, NULL);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "Can't compute boundary part of ImplicitMatrix-vector product: " << m_oclHelper.ErrorString(err);
        ASSERT_OR_THROW(sstr.str().c_str(), false);
    }

}

size_t ImplicitMatrix::fieldLength() const {
    return (size_t) mh_solverParams.xDim * mh_solverParams.yDim * mh_solverParams.zDim;
}


void ImplicitMatrix::ApplyBCToRHS(cl_mem xVct) const {
    if (!hasNonPeriodic(&mh_boundaryConditions))
        return;

    try {
        m_modifyRHStoBC->setArgument(3, xVct);
    } catch (std::exception &ec) {
        ASSERT_OR_THROW(ec.what(), false);
    }

    size_t glob_size[] = {(size_t) std::max(mh_solverParams.xDim, mh_solverParams.yDim),
                          (size_t) std::max(mh_solverParams.yDim, mh_solverParams.zDim)};
    cl_int err = m_oclHelper.EnqueueNDRangeKernel(m_modifyRHStoBC->getKernel(), 2, glob_size, NULL);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "Can't modify RHS according to BC: " << m_oclHelper.ErrorString(err);
        ASSERT_OR_THROW(sstr.str().c_str(), false);
    }
}
