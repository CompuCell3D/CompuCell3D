SET(SRCS
  BoundaryStrategy.cpp
  DefaultAlgorithm.cpp
  PeriodicBoundary.cpp
  ChengbangAlgorithm.cpp
  NoFluxBoundary.cpp
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Boundary ${SRCS} LINK_LIBRARIES BasicUtilsStatic Field3DStatic)
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Boundary ${SRCS} LINK_LIBRARIES BasicUtilsShared Field3DShared)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/CompuCell3D/Boundary .h
  AlgorithmFactory
  Algorithm
  BoundaryFactory
  Boundary
  BoundaryTypeDefinitions
  BoundaryStrategy
  BoundaryDLLSpecifier
  ChengbangAlgorithm
  DefaultAlgorithm
  NoFluxBoundary
  PeriodicBoundary
  CC3D_Boundary
  )
