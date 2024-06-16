#include "OpenCLHelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>
#include <stdexcept>

#include <CompuCell3D/CC3DExceptions.h>
#include <Logger/CC3DLogger.h>

using namespace CompuCell3D;

using std::endl;

OpenCLHelper::OpenCLHelper(int gpuDeviceIndex, int platformHint) {
    if (gpuDeviceIndex == -1)
        gpuDeviceIndex = 0;//TODO: check if we can find "the best" device

    int selectedPlatformInd;

    //this function is defined in util.cpp
    //it comes from the NVIDIA SDK example code
    cl_int err = GetPlatformID(platform, selectedPlatformInd, platformHint);
    //also comes from the NVIDIA SDK
    if (err != CL_SUCCESS)
        std::cout << "oclGetPlatformID: " << ErrorString(err) << std::endl;

    // Get the number of GPU devices available to the platform
    // we should probably expose the device type to the user
    // the other common option is CL_DEVICE_TYPE_CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (err != CL_SUCCESS)
        std::cout << "clGetDeviceIDs (get number of devices): " << ErrorString(err) << std::endl;
    ASSERT_OR_THROW("OpenCLHelper::ctor Can't use the requested device: there is no device with this index",
                    gpuDeviceIndex < (int) numDevices);
    std::cout << "\t" << numDevices << " device(s) available in platform #" << selectedPlatformInd << endl;

    /*if((size_t)gpuDeviceIndex>=numDevices){
        throw std::runtime_error("OpenCLHelper::ctor Can't use the requested device: there is no device with this index");
    }*/

    deviceUsed = gpuDeviceIndex;

    // Create the device list
    devices = new cl_device_id[numDevices];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    if (err != CL_SUCCESS)
        std::cout << "clGetDeviceIDs (create device list): " << ErrorString(err) << std::endl;


    //create the context
    context = clCreateContext(0, 1, &devices[deviceUsed], NULL, NULL, &err);
    assert(err == CL_SUCCESS);

    //create the command queue we will use to execute OpenCL commands
    commandQueue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
	assert(err==CL_SUCCESS);
	char devName[256];
	size_t len;
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_NAME,255,devName, &len);
    CC3D_Log(LOG_DEBUG) << "\tGPU device \""<<devName<<"\" selected";
	cl_ulong gpuMem;
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gpuMem), &gpuMem, &len);
    CC3D_Log(LOG_DEBUG) << "\tTotal GPU memory: "<<gpuMem/1024/1024<<"MB";

	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(gpuMem), &gpuMem, &len);
    CC3D_Log(LOG_DEBUG) << "\tMax GPU memory chunk to allocate: "<<gpuMem/1024/1024<<"MB";

	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, &len);
    CC3D_Log(LOG_DEBUG) << "\tMax work group size: "<<maxWorkGroupSize;
}


OpenCLHelper::~OpenCLHelper() {
    Finish();
    cl_int res;

    res = clReleaseCommandQueue(commandQueue);
    assert(res == CL_SUCCESS);

    res = clReleaseContext(context);
    assert(res == CL_SUCCESS);

    delete devices;

}

void OpenCLHelper::Finish() {
    clFinish(commandQueue);
}

cl_int OpenCLHelper::EnqueueNDRangeKernel(cl_kernel kernel,
                                          cl_uint work_dim,
                                          const size_t *global_work_size,
                                          const size_t *local_work_size
) const {
    return clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL,
                                  NULL);
}

//memory allocator
cl_mem OpenCLHelper::CreateBuffer(cl_mem_flags memFlags, size_t sizeInBytes, void const *hostPtr) const {
    cl_int retCode;
    if (hostPtr && !(memFlags & CL_MEM_COPY_HOST_PTR))
        memFlags |= CL_MEM_COPY_HOST_PTR;
    cl_mem res = clCreateBuffer(context, memFlags, sizeInBytes, const_cast<void *>(hostPtr), &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can not allocate GPU memory");
    return res;
}

cl_program OpenCLHelper::CreateProgramWithSource(cl_uint sourcesCount,
                                                 const char **kernel_source,
                                                 cl_int *errcode_ret) const {
    return clCreateProgramWithSource(context, sourcesCount, kernel_source, NULL, errcode_ret);
}

void OpenCLHelper::BuildExecutable(cl_program program) const {
    // Build the program executable
    CC3D_Log(LOG_DEBUG) << "building the program";
    // build the programming
    //err = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL);
    cl_int buildErr = clBuildProgram(program, 1, &devices[deviceUsed], NULL, NULL, NULL);
    CC3D_Log(LOG_DEBUG) << "clBuildProgram: " << ErrorString(buildErr);
    CC3D_Log(LOG_DEBUG) << deviceUsed<<" "<<numDevices;
    //if(err != CL_SUCCESS){
    cl_build_status build_status;
    cl_int err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status),
                                       &build_status, NULL);
    char *build_log;
    size_t ret_val_size;
    err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

    build_log = new char[ret_val_size + 1];
    err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

    build_log[ret_val_size] = '\0';
    CC3D_Log(LOG_DEBUG) << "BUILD LOG: " << std::endl << build_log;
    delete build_log;

    //}
    //if(buildErr != CL_SUCCESS)
    //	throw(std::runtime_error("error"));
    ASSERT_OR_THROW("Can not build the GPU program", buildErr == CL_SUCCESS);
    CC3D_Log(LOG_DEBUG) << "program built";
}


char *OpenCLHelper::FileContents(const char *filename, int *length) {
    FILE *f = fopen(filename, "r");
    void *buffer;

    if (!f) {
        CC3D_Log(LOG_ERROR) <<  "Unable to open " << filename << " for reading";
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(*length + 1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    ((char *) buffer)[*length] = '\0';

    return (char *) buffer;
}

bool OpenCLHelper::LoadProgram(const char *filePath[], size_t sourcesCount, cl_program &program) const {
    cl_int err;

    std::vector<int> len(sourcesCount);
    std::vector<const char *> kernel_source(sourcesCount);

    for (size_t i = 0; i < sourcesCount; ++i) {
        char *kernel_buff = FileContents(filePath[i], &len[i]);

        if (!kernel_buff) {
            std::stringstream sstr;
            sstr << "OpenCLHelper::LoadProgram: Can't load '" << filePath[i] << "' program";
            throw std::invalid_argument(sstr.str().c_str());
        }

        kernel_source[i] = kernel_buff;
    }



    // create the program
    program = CreateProgramWithSource(sourcesCount,
                                      &kernel_source[0], &err);

    CC3D_Log(LOG_DEBUG) << "clCreateProgramWithSource: " << ErrorString(err);

    if (err != CL_SUCCESS)
        return false;

    BuildExecutable(program);

    for (size_t i = 0; i < sourcesCount; ++i) {
        if (kernel_source[i])
            free((char *) kernel_source[i]);
    }


    return true;
}




////////////////////
//helper functions//
////////////////////


//NVIDIA's code follows
//license issues probably prevent you from using this, but shouldn't take long
//to reimplement
//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platform ID
//////////////////////////////////////////////////////////////////////////////
cl_int OpenCLHelper::GetPlatformID(cl_platform_id &clSelectedPlatformID, int &platformInd, int platfrormHint) {
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id *clPlatformIDs;
    clSelectedPlatformID = NULL;
    cl_uint i = 0;
    platformInd = -1;

    // Get OpenCL platform count
    cl_int ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS) {
        //shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        CC3D_Log(LOG_ERROR) << " Error " << ciErrNum << " in clGetPlatformIDs Call !!!";
        return -1000;
    } else {
        if (num_platforms == 0) {
            //shrLog("No OpenCL platform found!\n\n");
            CC3D_Log(LOG_ERROR) << "No OpenCL platform found!";
            return -2000;
        } else {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
                CC3D_Log(LOG_ERROR) << "Failed to allocate memory for cl_platform ID's!";
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
            CC3D_Log(LOG_DEBUG) << "Available platforms:";
            for (i = 0; i < num_platforms; ++i) {
                ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if (ciErrNum == CL_SUCCESS) {
                    CC3D_Log(LOG_DEBUG) << "platform " << i << ": " << chBuffer;
                    if (platfrormHint == -1 && strstr(chBuffer, "NVIDIA") != NULL) {
                        CC3D_Log(LOG_DEBUG) << "selected default NVIDIA platform (" << i << ")";
                        clSelectedPlatformID = clPlatformIDs[i];
                        platformInd = i;
                        break;
                    }
                }
            }

            if (platfrormHint != -1 && platfrormHint < (int) num_platforms) {
                CC3D_Log(LOG_DEBUG) << "Platform " << platfrormHint << " selection requested";
                clSelectedPlatformID = clPlatformIDs[platfrormHint];
                platformInd = platfrormHint;
            }

            // default to zeroeth platform if NVIDIA not found
            if (clSelectedPlatformID == NULL) {
                //shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                CC3D_Log(LOG_WARNING) << "WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!";
                clSelectedPlatformID = clPlatformIDs[0];
                platformInd = 0;
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}


// Helper function to get error string
// *********************************************************************
const char *OpenCLHelper::ErrorString(cl_int error) {
    static const char *errorString[] = {
            "CL_SUCCESS",
            "CL_DEVICE_NOT_FOUND",
            "CL_DEVICE_NOT_AVAILABLE",
            "CL_COMPILER_NOT_AVAILABLE",
            "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            "CL_OUT_OF_RESOURCES",
            "CL_OUT_OF_HOST_MEMORY",
            "CL_PROFILING_INFO_NOT_AVAILABLE",
            "CL_MEM_COPY_OVERLAP",
            "CL_IMAGE_FORMAT_MISMATCH",
            "CL_IMAGE_FORMAT_NOT_SUPPORTED",
            "CL_BUILD_PROGRAM_FAILURE",
            "CL_MAP_FAILURE",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "CL_INVALID_VALUE",
            "CL_INVALID_DEVICE_TYPE",
            "CL_INVALID_PLATFORM",
            "CL_INVALID_DEVICE",
            "CL_INVALID_CONTEXT",
            "CL_INVALID_QUEUE_PROPERTIES",
            "CL_INVALID_COMMAND_QUEUE",
            "CL_INVALID_HOST_PTR",
            "CL_INVALID_MEM_OBJECT",
            "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
            "CL_INVALID_IMAGE_SIZE",
            "CL_INVALID_SAMPLER",
            "CL_INVALID_BINARY",
            "CL_INVALID_BUILD_OPTIONS",
            "CL_INVALID_PROGRAM",
            "CL_INVALID_PROGRAM_EXECUTABLE",
            "CL_INVALID_KERNEL_NAME",
            "CL_INVALID_KERNEL_DEFINITION",
            "CL_INVALID_KERNEL",
            "CL_INVALID_ARG_INDEX",
            "CL_INVALID_ARG_VALUE",
            "CL_INVALID_ARG_SIZE",
            "CL_INVALID_KERNEL_ARGS",
            "CL_INVALID_WORK_DIMENSION",
            "CL_INVALID_WORK_GROUP_SIZE",
            "CL_INVALID_WORK_ITEM_SIZE",
            "CL_INVALID_GLOBAL_OFFSET",
            "CL_INVALID_EVENT_WAIT_LIST",
            "CL_INVALID_EVENT",
            "CL_INVALID_OPERATION",
            "CL_INVALID_GL_OBJECT",
            "CL_INVALID_BUFFER_SIZE",
            "CL_INVALID_MIP_LEVEL",
            "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";

}
