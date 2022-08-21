#include "OpenCLBuffer.h"
#include "OpenCLHelper.h"

using namespace CompuCell3D;

OpenCLBuffer::OpenCLBuffer(OpenCLHelper const &helper, cl_mem_flags memFlags, size_t sizeInBytes, void const *hostPtr) {
    m_buffer = helper.CreateBuffer(memFlags, sizeInBytes, hostPtr);
}

OpenCLBuffer::OpenCLBuffer(cl_mem memBuff) : m_buffer(memBuff) {}//take ownership


OpenCLBuffer::~OpenCLBuffer(void) {
    clReleaseMemObject(m_buffer);
}

