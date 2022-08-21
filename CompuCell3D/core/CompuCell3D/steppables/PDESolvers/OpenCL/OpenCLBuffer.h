#pragma once

//RAII wrapper aroung cl_mem

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif


namespace CompuCell3D {

    class OpenCLHelper;

    class OpenCLBuffer {
        cl_mem m_buffer;
    public:
        OpenCLBuffer(OpenCLHelper const &helper, cl_mem_flags memFlags, size_t sizeInBytes, void const *hostPtr);

        //take ownership
        explicit OpenCLBuffer(cl_mem memBuff);

        ~OpenCLBuffer(void);

        cl_mem const &buffer() const { return m_buffer; }
    };

}