#include "LinearSolver.h"
#include <CompuCell3D/CC3DExceptions.h>

#include "../GPUSolverParams.h"
#include "ImplicitMatrix.h"
#include <viennacl/linalg/cg.hpp>
#include "Helper.h"

#include "../MyTime.h"


using namespace CompuCell3D;

LinearSolver::LinearSolver(OpenCLHelper const &oclHelper,
                           std::vector <UniSolverParams> const &solverParams, cl_mem const &d_cellTypes,
                           GPUBoundaryConditions const &boundaryConditions,
                           unsigned char fieldsCount, std::string const &pathToKernels) : Solver(oclHelper,
                                                                                                 solverParams,
                                                                                                 d_cellTypes,
                                                                                                 boundaryConditions,
                                                                                                 fieldsCount,
                                                                                                 pathToKernels) {

}


viennacl::vector<float> const &
LinearSolver::NewField(float dt, viennacl::vector<float> const &v_oldField, NLSParams const &nlsParams) {
    setTimeStep(dt);

    MyTime::Time_t stepBT = MyTime::CTime();
    viennacl::linalg::cg_tag solver_tag = viennacl::linalg::cg_tag(nlsParams.linear_.tol_,
                                                                   nlsParams.linear_.maxIterations_);

    applyBCToRHS(v_oldField);

    getOutVector() = viennacl::linalg::solve(*this,
                                             v_oldField,
                                             solver_tag);
    addSolvingTime(MyTime::ElapsedTime(stepBT, MyTime::CTime()));

    //analyze("New fields: ", mv_outputField, dim(), m_fieldsCount);
    return getOutVector();
}

viennacl::vector<float> const &LinearSolver::prod(viennacl::vector<float> const &v_update) const {
    size_t totalLength = getFieldLength();
    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        cl_buffer_region subBufferInfo = {i * totalLength * sizeof(float), totalLength * sizeof(float)};
        cl_int retCode;
        cl_mem subBuffer = clCreateSubBuffer(v_update.handle().opencl_handle().get(), CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION, &subBufferInfo, &retCode);
        ASSERT_OR_THROW("Can't make an update subbuffer", retCode == CL_SUCCESS);

        //solving current field
        prodField(i, subBuffer);

        clReleaseMemObject(subBuffer);
    }


    return getOutVector();
}
