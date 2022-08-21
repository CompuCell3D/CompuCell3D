#include "OpenCLKernel.h"
#include <sstream>
#include "OpenCLBuffer.h"
#include "OpenCLHelper.h"

#include <stdexcept>

using namespace CompuCell3D;

OpenCLKernel::OpenCLKernel(cl_program program, const char *kernelName) : m_kernelName(kernelName) {
    cl_int err;
    m_kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "OpenCLHelper ctor: Can't create: '" << kernelName << "' kernel: " << OpenCLHelper::ErrorString(err);
        //throw std::runtime_error(sstr.str().c_str());
        throw std::runtime_error(sstr.str().c_str());
    }
}

OpenCLKernel::~OpenCLKernel(void) {
    clReleaseKernel(m_kernel);
}

void OpenCLKernel::CheckForArgError(cl_int err, int argInd) const {
    if (err != CL_SUCCESS) {
        std::stringstream sstr;
        sstr << "OpenCLKernel::setArgument Can't pass argument #" << argInd << " to '" << m_kernelName
             << "' kernel. Error: " << OpenCLHelper::ErrorString(err);
        throw std::runtime_error(sstr.str().c_str());
    }
}

//bind device memory to an argument

void OpenCLKernel::setArgument(int argInd, cl_mem deviceMem) const {
    cl_int err = clSetKernelArg(m_kernel, argInd, sizeof(cl_mem), &deviceMem);
    CheckForArgError(err, argInd);
}

/*void OpenCLKernel::setArgument(int argInd, OpenCLBuffer &buffer){
	cl_int err=clSetKernelArg(m_kernel, argInd, sizeof(cl_mem), &buffer.buffer());
	CheckForArgError(err, argInd);
}*/

template<typename T>
void OpenCLKernel::setArgument(int argInd, T value) const {
    cl_int err = clSetKernelArg(m_kernel, argInd, sizeof(T), &value);
    CheckForArgError(err, argInd);
}

cl_kernel const &
OpenCLKernel::getKernel() const {
    return m_kernel;
}

//Explicit instantiation
//HINT: Add other types here
template
void OpenCLKernel::setArgument<float>(int argInd, float value) const;

template
void OpenCLKernel::setArgument<int>(int argInd, int value) const;

template
void OpenCLKernel::setArgument<cl_int4>(int argInd, cl_int4 value) const;
