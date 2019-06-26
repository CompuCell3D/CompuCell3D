#MESSAGE(STATUS "From OpenCL ::***OpenCL headers found at " ${OPENCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})

IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
ENDIF(MSVC)


SET (HEADER_FILES
	FlexibleDiffusionSolverFE_GPU_OpenCL.h
	OpenCLHelper.h
)
	
SET (KERNEL_FILES
  DiffusionKernel.cl
  common.cl
  ImplicitMatrix.cl
  )
  
SOURCE_GROUP(Kernels FILES
	${KERNEL_FILES})
	
#message(${PROJECT_SOURCE_DIR})

ADD_DEFINITIONS(-DPDESOLVERSGPU_EXPORT_COMPILER_DEFINITION)
ADD_LIBRARY(FlexibleDiffusionSolverFE_OpenCL SHARED 
	${HEADER_FILES}
	${KERNEL_FILES}
	FlexibleDiffusionSolverFE_GPU_OpenCL.cpp

	#I need an acces to ASSERT_OR_THROW macro
	#I don't really like includin this source file here. It might be better to compile it against the appropriate dll
	#However, this solver is abandoned anyway
	${PROJECT_SOURCE_DIR}/core/BasicUtils/BasicException.cpp
	OpenCLHelper.cpp
)

TARGET_LINK_LIBRARIES(FlexibleDiffusionSolverFE_OpenCL
	${OPENCL_LIBRARIES}) 

INSTALL_TARGETS(${COMPUCELL3D_INSTALL_STEPPABLE_DIR} RUNTIME_DIRECTORY ${COMPUCELL3D_INSTALL_STEPPABLE_DIR} FlexibleDiffusionSolverFE_OpenCL)
INSTALL(FILES 
	${KERNEL_FILES} 
	../GPUSolverParams.h
	GPUBoundaryConditions.h
	DESTINATION ${COMPUCELL3D_INSTALL_STEPPABLE_DIR_FULL_PATH}/OpenCL)