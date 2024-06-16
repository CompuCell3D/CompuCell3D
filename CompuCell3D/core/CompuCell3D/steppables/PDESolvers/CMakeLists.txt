IF(COMPUCELL3D_TEST)
  INCLUDE_DIRECTORIES(".")
  ADD_SUBDIRECTORY(Testing)
ENDIF(COMPUCELL3D_TEST)

IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
ENDIF(MSVC)

#enable looking for the FindEigen3.cmake and FindOpenCL.cmake (if requested) files in the source directory
MESSAGE(STATUS ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

#for the implicit diffusion solver
find_package(Eigen3 REQUIRED)
SET(INCLUDE_DIRS $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)

SET (HEADER_FILES
  DiffSecrData.h
  DiffusableGraph.h
  ReactionDiffusionSolverFE_SavHog.h
  ReactionDiffusionSolverFE.h
  ReactionDiffusionSolverFVM.h
  FlexibleReactionDiffusionSolverFE.h
  AdvectionDiffusionSolverFE.h
  Diffusable.h
  DiffusableVector2D.h
  FastDiffusionSolver2DFE.h
  SteadyStateDiffusionSolver2D.h
  SteadyStateDiffusionSolver.h
  DiffusableVector.h
  DiffusableVectorCommon.h
  DiffusableVectorFortran.h
  hpppdesolvers
  FlexibleDiffusionSolverFE.h
  # FiPySolver.h
  DiffusionSolverFE.h
  DiffusionSolverFE_CPU.h
  DiffusionSolverFE_CPU_Implicit.h
  FlexibleDiffusionSolverADE.h
  KernelDiffusionSolver.h
  GPUSolverParams.h
  FluctuationCompensator.h
  ${OPENCL_HDRS}
)

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


# ${OpenCL_LIBRARIES}
IF(NOT NO_OPENCL)
	MESSAGE(STATUS "Looking for OpenCL...")
	FIND_PACKAGE(OpenCL)
	
	IF(OPENCL_FOUND)
        message("FOUND OPEN CL")
#        IF(NOT WIN32)
#                SET(CMAKE_CXX_FLAGS "-std=c++0x")
#        ENDIF(NOT WIN32)
		MESSAGE(STATUS "OpenCL headers found at " ${OpenCL_INCLUDE_DIRS})
		MESSAGE(STATUS "OpenCL library: " ${OpenCL_LIBRARIES})
		ADD_SUBDIRECTORY(OpenCL)
		ADD_DEFINITIONS(-DPDESOLVERSGPU_EXPORT_COMPILER_DEFINITION -DVIENNACL_WITH_OPENCL)
		# INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
		# include_directories(${CMAKE_SOURCE_DIR}/core/ViennaCL)
    LIST(APPEND INCLUDE_DIRS 
      ${OpenCL_INCLUDE_DIRS}
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/core/ViennaCL>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/ViennaCL>
    )
		SET(OpenCLEnabled 1)
		SET(OPENCL_HDRS 
			#DiffusableVectorGPU.h
			#FlexibleDiffusionSolverFE_OpenCL.h
    )
		SET(OPENCL_SRCS
#			OpenCL/FlexibleDiffusionSolverFE_OpenCL.cpp
			OpenCL/DiffusionSolverFE_OpenCL.cpp
#			OpenCL/DiffusionSolverFE_OpenCL_Implicit.cpp
			OpenCL/ReactionDiffusionSolverFE_OpenCL_Implicit.cpp
			OpenCL/OpenCLHelper.cpp
			OpenCL/OpenCLKernel.cpp
			OpenCL/OpenCLBuffer.cpp
			OpenCL/ReactionKernels.cpp
#			OpenCL/NonLinear.cpp
			OpenCL/ImplicitMatrix.cpp
                        OpenCL/NonlinearSolver.cpp
			OpenCL/LinearSolver.cpp
			OpenCL/Solver.cpp
		)	
		# SET(OPENCL_LIBS FlexibleDiffusionSolverFE_OpenCL ${OpenCL_LIBRARIES})
        SET(OPENCL_LIBS ${OpenCL_LIBRARIES})
		SET(HEADER_FILES
			${HEADER_FILES}
#			FlexibleDiffusionSolverFE_GPU_Device.h
#			FlexibleDiffusionSolverFE_GPU.h
#			FlexibleDiffusionSolverFE_GPU.hpp
			OpenCL/DiffusionSolverFE_OpenCL.h
#			OpenCL/DiffusionSolverFE_OpenCL_Implicit.h
			OpenCL/ReactionDiffusionSolverFE_OpenCL_Implicit.h
			OpenCL/OpenCLHelper.h
			OpenCL/OpenCLKernel.h
			OpenCL/OpenCLBuffer.h
			OpenCL/ReactionKernels.h
			OpenCL/ImplicitMatrix.h
#			OpenCL/NonLinear.h
			OpenCL/NonlinearSolver.h
			OpenCL/DiffusableVectorRD.hpp
			OpenCL/OCLNeighbourIndsInfo.h
			OpenCL/Solver.h
			OpenCL/LinearSolver.h
#			OpenCL/ArrayRDOpenCLImpl.hpp
			${OPENCL_HDRS}
    )
		
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
LIST(APPEND INCLUDE_DIRS $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)


message("OPENMP FLAGS " ${OpenMP_CXX_FLAGS})

ADD_COMPUCELL3D_STEPPABLE(PDESolvers
  AdvectionDiffusionSolverFE.cpp
  FastDiffusionSolver2DFE.cpp
  DiffusionSolverFE.cpp
  DiffusionSolverFE_CPU.cpp
  DiffusionSolverFE_CPU_Implicit.cpp
  SteadyStateDiffusionSolver2D.cpp
  SteadyStateDiffusionSolver.cpp
  PDESolversProxy.cpp
  DiffSecrData.cpp  
  FlexibleDiffusionSolverFE.cpp
  # FiPySolver.cpp
  FlexibleDiffusionSolverADE.cpp
  KernelDiffusionSolver.cpp
  ReactionDiffusionSolverFE_SavHog.cpp
  ReactionDiffusionSolverFE.cpp
  ReactionDiffusionSolverFVM.cpp
  FlexibleReactionDiffusionSolverFE.cpp
  FluctuationCompensator.cpp
  HW3CRT.c 
  HWSCRT.c
#  ${CUDA_SRCS}
  ${OPENCL_SRCS}
  ${HEADER_FILES}

  LINK_LIBRARIES 
  ${STEPPABLE_DEPENDENCIES} 
  ${OPENCL_LIBS} 
#  ${CUDA_LIBS} 
  cc3d::PixelTracker 
  cc3d::NeighborTracker 
  cc3d::BoxWatcher 
  cc3d::CellType 

  EXTRA_COMPILER_FLAGS 
  ${OpenMP_CXX_FLAGS}
)

TARGET_INCLUDE_DIRECTORIES(${PDESolversLib} PUBLIC ${INCLUDE_DIRS})

ADD_COMPUCELL3D_STEPPABLE_HEADERS(PDESolvers ${HEADER_FILES})
