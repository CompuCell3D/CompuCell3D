#include "OpenCLHelper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cassert>
#include <sstream>

#include <BasicUtils/BasicException.h>

using namespace CompuCell3D;

using std::cerr;
using std::endl;
using std::stringstream;

OpenCLHelper::OpenCLHelper(int gpuDeviceIndex){
	if(gpuDeviceIndex==-1)
		 gpuDeviceIndex=0;//TODO: check if we can find "the best" device

    //this function is defined in util.cpp
    //it comes from the NVIDIA SDK example code
    cl_int err = GetPlatformID(&platform);
    //also comes from the NVIDIA SDK
    std::cout<<"oclGetPlatformID: "<<ErrorString(err)<<std::endl;

    // Get the number of GPU devices available to the platform
    // we should probably expose the device type to the user
    // the other common option is CL_DEVICE_TYPE_CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    std::cout<<"clGetDeviceIDs (get number of devices): "<<ErrorString(err)<<std::endl;
	stringstream errStr;
	errStr<<"OpenCLHelper::ctor Can't use the requested device: there is no device with this index ("<<gpuDeviceIndex<<")";
    ASSERT_OR_THROW(errStr.str().c_str() ,gpuDeviceIndex<(int)numDevices);

	/*if((size_t)gpuDeviceIndex>=numDevices){
		throw std::runtime_error("OpenCLHelper::ctor Can't use the requested device: there is no device with this index");
	}*/

    deviceUsed=gpuDeviceIndex;
	
    // Create the device list
    devices = new cl_device_id [numDevices];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    std::cout<<"clGetDeviceIDs (create device list): "<<ErrorString(err)<<std::endl;
 
    //create the context
    context = clCreateContext(0, 1, &devices[deviceUsed], NULL, NULL, &err);
	assert(err==CL_SUCCESS);

    //create the command queue we will use to execute OpenCL commands
    commandQueue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
	assert(err==CL_SUCCESS);
	char devName[256];
	size_t len;
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_NAME,255,devName, &len);
	cerr<<"GPU device \""<<devName<<"\" selected\n";
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, &len);
	cerr<<"Max work group size: "<<maxWorkGroupSize<<endl;
	cl_ulong gpuMem;
	clGetDeviceInfo(devices[deviceUsed], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gpuMem), &gpuMem, &len);
	cerr<<"GPU memory: "<<gpuMem<<endl;
}


OpenCLHelper::~OpenCLHelper(){
	Finish();
	cl_int res;

	res=clReleaseCommandQueue(commandQueue);
	assert(res==CL_SUCCESS);

	res=clReleaseContext(context);
	assert(res==CL_SUCCESS);

	delete devices;

}

void OpenCLHelper::Finish(){
	clFinish(commandQueue);
}

cl_int OpenCLHelper::EnqueueNDRangeKernel(cl_kernel kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size
                       )
{
	return clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

//memory allocator
cl_mem OpenCLHelper::CreateBuffer(cl_mem_flags memFlags, size_t sizeInBytes)const{
	cl_int retCode;
	cl_mem res=clCreateBuffer(context, memFlags, sizeInBytes, NULL, &retCode);
	ASSERT_OR_THROW("Can not allocate GPU memory", retCode==CL_SUCCESS);
	return res;
}

cl_program OpenCLHelper::CreateProgramWithSource(cl_uint sourcesCount,
                          const char ** kernel_source,
                          cl_int * errcode_ret)
{
	return clCreateProgramWithSource(context, sourcesCount, kernel_source, NULL, errcode_ret);
}

void OpenCLHelper::BuildExecutable(cl_program program)
{
    // Build the program executable

    printf("building the program\n");
    // build the program
    //err = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL);
	cl_int buildErr = clBuildProgram(program, 1, &devices[deviceUsed], NULL, NULL, NULL);
    printf("clBuildProgram: %s\n", ErrorString(buildErr));
	cerr<<deviceUsed<<" "<<numDevices<<endl;
    //if(err != CL_SUCCESS){
        cl_build_status build_status;
        cl_int err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
		char *build_log;
        size_t ret_val_size;
        err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
		
        build_log = new char[ret_val_size+1];
        err = clGetProgramBuildInfo(program, devices[deviceUsed], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
		
        build_log[ret_val_size] = '\0';
        printf("BUILD LOG: \n %s", build_log);
		delete build_log;
		
    //}
	//if(buildErr != CL_SUCCESS)
	//	throw(std::runtime_error("error"));
	ASSERT_OR_THROW("Can not build the GPU program", buildErr==CL_SUCCESS);
    printf("program built\n");
}


char *OpenCLHelper::FileContents(const char *filename, int *length)
{
    FILE *f = fopen(filename, "r");
    void *buffer;

    if (!f) {
        fprintf(stderr, "Unable to open %s for reading\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    ((char*)buffer)[*length] = '\0';

    return (char*)buffer;
}

bool OpenCLHelper::LoadProgram(const char *filePath[], size_t sourcesCount, cl_program &program)
{
    cl_int err;
	
	int *len=new int[sourcesCount];
    const char **kernel_source=new const char *[sourcesCount];

	for(size_t i=0; i<sourcesCount; ++i){
		char *kernel_buff= FileContents(filePath[i], &len[i]);
		
		kernel_source[i]=kernel_buff;
	}

    


    // create the program
    program = CreateProgramWithSource(sourcesCount,
                                        kernel_source, &err);

    printf("clCreateProgramWithSource: %s\n", ErrorString(err));

	if(err!=CL_SUCCESS)
		return false;

    BuildExecutable(program);

	for(size_t i=0; i<sourcesCount; ++i){
		if(kernel_source[i])
			free((char *)kernel_source[i]);
	}
	   
	delete len;
	delete kernel_source; 

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
cl_int OpenCLHelper::GetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;
    cl_uint i = 0;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
        //shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
        return -1000;
    }
    else
    {
        if(num_platforms == 0)
        {
            //shrLog("No OpenCL platform found!\n\n");
            printf("No OpenCL platform found!\n\n");
            return -2000;
        }
        else
        {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
            {
                printf("Failed to allocate memory for cl_platform ID's!\n\n");
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            printf("Available platforms:\n");
            for(i = 0; i < num_platforms; ++i)
            {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS)
                {
                    printf("platform %d: %s\n", i, chBuffer);
                    if(strstr(chBuffer, "NVIDIA") != NULL)
                    {
                        printf("selected platform %d\n", i);
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL)
            {
                //shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                //printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                printf("selected platform: %d\n", 0);
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}


// Helper function to get error string
// *********************************************************************
const char* OpenCLHelper::ErrorString(cl_int error)
{
    static const char* errorString[] = {
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