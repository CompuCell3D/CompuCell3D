#pragma once

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif


namespace CompuCell3D {

    class OpenCLBuffer;

    class OpenCLKernel {
        cl_kernel m_kernel;
        const char *m_kernelName;
    public:
        OpenCLKernel(cl_program program, const char *kernelName);

        ~OpenCLKernel(void);

        //bind device memory to an argument
        void setArgument(int argInd, cl_mem deviceMem) const;

        //bind device memory to an argument
        //void setArgument(int argInd, OpenCLBuffer &buffer);

        //bind value to an argument
        template<typename T>
        void setArgument(int argInd, T value) const;

        cl_kernel const &getKernel() const;

    private:
        void CheckForArgError(cl_int err, int argInd) const;
    };

}