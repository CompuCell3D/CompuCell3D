#include "Solver.h"

#include <CompuCell3D/CC3DExceptions.h>
#include "ImplicitMatrix.h"
#include "../GPUSolverParams.h"
using namespace CompuCell3D;

Dim3D Solver::getDim() const {
    assert(!m_ims.empty());
    return m_ims.front()->domainSize();
}

Solver::Solver(OpenCLHelper const &oclHelper, 
		std::vector<UniSolverParams> const &solverParams, cl_mem const &d_cellTypes, GPUBoundaryConditions const &boundaryConditions,
		unsigned char fieldsCount, std::string const &pathToKernels):
m_fieldsCount(fieldsCount), mv_outputField(fieldLength(&solverParams[0])*fieldsCount),m_oclHelper(oclHelper), m_linearST(0)
{
	ASSERT_OR_THROW("These two OpenCL contexts must be equal", m_oclHelper.getContext()==viennacl::ocl::current_context().handle().get());

    int fl = fieldLength(&solverParams[0]);//length of field, in elements

    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        cl_mem outBuffer = mv_outputField.handle().opencl_handle().get();

        cl_buffer_region subBufferInfo = {i * fl * sizeof(float), fl * sizeof(float)};
        cl_int retCode;
        cl_mem subBuffer = clCreateSubBuffer(outBuffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &subBufferInfo,
                                             &retCode);
        ASSERT_OR_THROW("Can't make an output subbuffer", retCode == CL_SUCCESS);
        m_outFieldSubBuffers.push_back(subBuffer);

        m_ims.push_back(new ImplicitMatrix(oclHelper, solverParams[i], d_cellTypes, subBuffer, boundaryConditions,
                                           pathToKernels));

    }
}

Solver::~Solver() {
    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        clReleaseMemObject(m_outFieldSubBuffers[i]);
    }

    for (size_t i = 0; i < m_ims.size(); ++i) {
        delete m_ims[i];
    }
}

OpenCLHelper const &Solver::getOCLHelper() const {
    return m_oclHelper;
}

void Solver::setTimeStep(float dt) {
    m_dt = dt;
    for (unsigned char i = 0; i < m_fieldsCount; ++i)
        m_ims[i]->setdt(dt);

    //m_timeStepSet=true;
}

cl_mem Solver::prodField(int i, cl_mem v) const {
    ASSERT_OR_THROW("Wrong index", (size_t) i < m_ims.size());
    return m_ims[i]->prod(v);
}

void Solver::applyBCToRHSField(int i, cl_mem v) const {
    ASSERT_OR_THROW("Wrong index", (size_t) i < m_ims.size());
    m_ims[i]->ApplyBCToRHS(v);
}

int Solver::getFieldLength() const {
    return m_ims[0]->fieldLength();
}

void Solver::applyBCToRHS(viennacl::vector<float> const &rhs) const {
    size_t totalLength = getFieldLength();
    for (unsigned char i = 0; i < getFieldsCount(); ++i) {
        cl_buffer_region subBufferInfo = {i * totalLength * sizeof(float), totalLength * sizeof(float)};
        cl_int retCode;
        cl_mem subBuffer = clCreateSubBuffer(rhs.handle().opencl_handle().get(), CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION, &subBufferInfo, &retCode);
        ASSERT_OR_THROW("Can't make an old field subbuffer", retCode == CL_SUCCESS);
        applyBCToRHSField(i, subBuffer);
        clReleaseMemObject(subBuffer);
    }
}