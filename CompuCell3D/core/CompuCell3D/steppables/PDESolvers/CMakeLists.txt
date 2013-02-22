IF(COMPUCELL3D_TEST)
  INCLUDE_DIRECTORIES(".")
  ADD_SUBDIRECTORY(Testing)
ENDIF(COMPUCELL3D_TEST)

IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
ENDIF(MSVC)

SET (HEADER_FILES
  hpppdesolvers
  GPUSolverParams.h
  ${OPENCL_HDRS})


# SET (HEADER_FILES
  # DiffSecrData.h
  # DiffusableGraph.h
  # ReactionDiffusionSolverFE_SavHog.h
  # ReactionDiffusionSolverFE.h
  # AdvectionDiffusionSolverFE.h
  # Diffusable.h
  # DiffusableVector2D.h
  # FastDiffusionSolver2DFE.h
  # SteadyStateDiffusionSolver2D.h
  # SteadyStateDiffusionSolver.h
  # DiffusableVector.h
  # DiffusableVectorCommon.h
  # DiffusableVectorFortran.h
  # hpppdesolvers
  # FlexibleDiffusionSolverFE.h
  # FiPySolver.h
  # DiffusionSolverFE.h
  # DiffusionSolverFE_CPU.h
  # FlexibleDiffusionSolverADE.h
  # KernelDiffusionSolver.h
  # GPUSolverParams.h
  # ${OPENCL_HDRS})

# SET(CUDA_LIBS )    
# SET(CUDA_SRCS )
# 
# #TODO: extract common parts for OpenCL and CUDA branches
# if(NOT NO_CUDA)
#     #MESSAGE(STATUS "Looking for CUDA...")
#     FIND_PACKAGE(CUDA)  
#     IF(CUDA_CUDA_LIBRARY)
# 		MESSAGE(STATUS "CUDA headers found at " ${CUDA_INCLUDE_DIRS})
# 		MESSAGE(STATUS "CUDA library: " ${CUDA_LIBRARIES})
# 		INCLUDE(FindCUDA)
# 		#MESSAGE(STATUS ${CUDA_TOOLKIT_INCLUDE})
# 		ADD_SUBDIRECTORY(CUDA)
# 
# 		#SET(CUDA_SRCS FlexibleDiffusionSolverFE_GPU.cpp)    
# 		
# 		SET(CUDA_LIBS FlexibleDiffusionSolverFE_CUDA)    
# 		
# 		ADD_DEFINITIONS(-DPDESOLVERSGPU_EXPORT_COMPILER_DEFINITION)
# 		
# 		SET(HEADER_FILES
# 			${HEADER_FILES}
# 			#DiffusableVectorGPU.h
# 			FlexibleDiffusionSolverFE_GPU_Device.h
# 			FlexibleDiffusionSolverFE_GPU.h
# 			FlexibleDiffusionSolverFE_GPU.hpp)
# 		
# 		INCLUDE_DIRECTORIES("c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.1/C/common/inc")    
# 		INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
# 		INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/CUDA)
# 		INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
# 
# 		SET(CudaEnabled 1)
# 	ELSE(CUDA_CUDA_LIBRARY)
# 		SET(CudaEnabled 0) 
# 		MESSAGE(STATUS "No CUDA found")
#     ENDIF(CUDA_CUDA_LIBRARY)
# 	
# else(NOT NO_CUDA)
#     SET(CudaEnabled 0)    
# endif(NOT NO_CUDA)

###begin OpenCL stuff

SET(OPENCL_LIBS)    
SET(OPENCL_SRCS)
SET(OPENCL_HDRS)

IF(NOT NO_OPENCL)
	MESSAGE(STATUS "Looking for OpenCL...")
	#enable looking for the FindOpenCL.cmake file in the source directory
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
	FIND_PACKAGE(OpenCL)
	
	IF(OPENCL_FOUND)
		MESSAGE(STATUS "OpenCL headers found at " ${OPENCL_INCLUDE_DIRS})
		MESSAGE(STATUS "OpenCL library: " ${OPENCL_LIBRARIES})
		ADD_SUBDIRECTORY(OpenCL)
		ADD_DEFINITIONS(-DPDESOLVERSGPU_EXPORT_COMPILER_DEFINITION)
		INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})
		SET(OpenCLEnabled 1)
		SET(OPENCL_HDRS 
			#DiffusableVectorGPU.h
			#FlexibleDiffusionSolverFE_OpenCL.h
			)
		SET(OPENCL_SRCS
#			OpenCL/FlexibleDiffusionSolverFE_OpenCL.cpp
			OpenCL/DiffusionSolverFE_OpenCL.cpp
			OpenCL/OpenCLHelper.cpp
		)	
		SET(OPENCL_LIBS
			FlexibleDiffusionSolverFE_OpenCL)
		SET(HEADER_FILES
			${HEADER_FILES}
#			FlexibleDiffusionSolverFE_GPU_Device.h
#			FlexibleDiffusionSolverFE_GPU.h
#			FlexibleDiffusionSolverFE_GPU.hpp
			OpenCL/DiffusionSolverFE_OpenCL.h
			OpenCL/OpenCLHelper.h
			${OPENCL_HDRS})
		
	ELSE(OPENCL_FOUND)
		MESSAGE('No OpenCL found')
		SET(OpenCLEnabled 0)
	ENDIF(OPENCL_FOUND)
	
ELSE (NOT NO_OPENCL)
	MESSAGE(STATUS "OpenCL disabled")
	SET(OpenCLEnabled 0)
ENDIF(NOT NO_OPENCL)
###end OpenCL stuff

# configure a header file to pass some of the CMake settings
# to the source code 
CONFIGURE_FILE (
	"${CMAKE_CURRENT_SOURCE_DIR}/GPUEnabled.h.in"
	"${PROJECT_BINARY_DIR}/GPUEnabled.h"
)

# add the binary tree to the search path for include files
# so that we will find CudaEnabled.h
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")


message("OPENMP FLAGS " ${OpenMP_CXX_FLAGS})

# message(opencl_srcs ${OPENCL_SRCS})

ADD_COMPUCELL3D_STEPPABLE(PDESolvers ${OPENCL_SRCS} ${HEADER_FILES} LINK_LIBRARIES ${STEPPABLE_DEPENDENCIES} ${OPENCL_LIBS} 
#  ${CUDA_LIBS} 
  EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
)

# ADD_COMPUCELL3D_STEPPABLE_HEADERS(PDESolvers
  # ${HEADER_FILES}
  # )



# ADD_COMPUCELL3D_STEPPABLE(PDESolvers
  # AdvectionDiffusionSolverFE.cpp
  # FastDiffusionSolver2DFE.cpp
  # DiffusionSolverFE.cpp
  # DiffusionSolverFE_CPU.cpp
  # SteadyStateDiffusionSolver2D.cpp
  # SteadyStateDiffusionSolver.cpp
  # PDESolversProxy.cpp
  # DiffSecrData.cpp  
  # FlexibleDiffusionSolverFE.cpp
  # # FiPySolver.cpp
  # FlexibleDiffusionSolverADE.cpp
  # KernelDiffusionSolver.cpp
  # ReactionDiffusionSolverFE_SavHog.cpp
  # ReactionDiffusionSolverFE.cpp
  # HW3CRT.c 
  # HWSCRT.c
# #  ${CUDA_SRCS}
  # ${OPENCL_SRCS}
  # ${HEADER_FILES}

# LINK_LIBRARIES ${STEPPABLE_DEPENDENCIES} ${OPENCL_LIBS} 
# #  ${CUDA_LIBS} 
  # EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS}
# )

# ADD_COMPUCELL3D_STEPPABLE_HEADERS(PDESolvers
  # ${HEADER_FILES}
  # )
