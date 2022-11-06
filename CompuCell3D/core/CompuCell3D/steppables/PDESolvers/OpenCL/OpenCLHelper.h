#ifndef OPENCL_HELPER_H
#define OPENCL_HELPER_H


// 2012 Mitja:
// #include <CL/opencl.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else

#include <CL/opencl.h>

#endif


#include <cassert>
#include <Logger/CC3DLogger.h>
namespace CompuCell3D {

    class OpenCLHelper {
        cl_context context;//device's context
        cl_command_queue commandQueue;
        cl_uint numDevices;
        cl_device_id *devices;
        cl_uint deviceUsed;
        cl_platform_id platform;

        size_t maxWorkGroupSize;

        static cl_int GetPlatformID(cl_platform_id &clSelectedPlatformID, int &platformInd, int platfrormHint = -1);

        void BuildExecutable(cl_program program) const;

        cl_program CreateProgramWithSource(cl_uint           /* count */,
                                           const char **     /* strings */,
                                           cl_int *          /* errcode_ret */) const;

    public:
        explicit OpenCLHelper(int gpuDeviceIndex, int platformHint = -1);

        ~OpenCLHelper();

        static const char *ErrorString(cl_int error);

        static char *FileContents(const char *filename, int *length);

        //simple, synchronous write (from host to device)
        template<class T>
        cl_int WriteBuffer(cl_mem buffer, T const *h_arr, size_t arrLen) const {
            cl_int res = clEnqueueWriteBuffer(commandQueue, buffer, CL_TRUE, 0, sizeof(T) * arrLen, h_arr, 0, NULL,
                                              NULL);
            CC3D_Log(LOG_DEBUG) << "sizeof(T)="<<sizeof(T)<<",  arrLen="<<arrLen;
            assert(res == CL_SUCCESS);
            return res;
        }

        //simple, synchronous copying (on device)
        template<class T>
        cl_int CopyBuffer(cl_mem src_buffer, cl_mem dst_buffer, size_t arrLen) const {
            cl_int res = clEnqueueCopyBuffer(commandQueue, src_buffer, dst_buffer, 0, 0, sizeof(T) * arrLen, 0, NULL,
                                             NULL);
            CC3D_Log(LOG_DEBUG) << "sizeof(T)="<<sizeof(T)<<",  arrLen="<<arrLen;
            assert(res == CL_SUCCESS);
            clEnqueueBarrier(commandQueue);//TODO: can be optimize with events probably
            return res;
        }

        //simple, synchronous read (from device to host)
        template<class T>
        cl_int ReadBuffer(cl_mem buffer, T *h_arr, size_t arrLen) const {
            cl_int res = clEnqueueReadBuffer(commandQueue, buffer, CL_TRUE, 0, sizeof(T) * arrLen, h_arr, 0, NULL,
                                             NULL);
            assert(res == CL_SUCCESS);
            return res;
        }

        size_t getMaxWorkGroupSize() const { return maxWorkGroupSize; }

        cl_int EnqueueNDRangeKernel(cl_kernel        /* kernel */,
                                    cl_uint          /* work_dim */,
                                    const size_t *   /* global_work_size */,
                                    const size_t *   /* local_work_size */
        ) const;

        cl_mem CreateBuffer(cl_mem_flags memFlags, size_t sizeInBytes, void const *hostPtr = NULL) const;

        bool LoadProgram(const char *filePath[], size_t sourcesCount, cl_program &program) const;

        void Finish();

        cl_context getContext() const { return context; }

        cl_device_id getDevice() const { return devices[deviceUsed]; }

        cl_command_queue getCommandQueue() const { return commandQueue; }
    };

}//namespace CompuCell3D 

#endif