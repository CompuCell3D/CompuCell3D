SET(SRCS
  BoundaryStrategy.cpp
  Boundary.cpp
  DefaultAlgorithm.cpp
  PeriodicBoundary.cpp
  ChengbangAlgorithm.cpp
  NoFluxBoundary.cpp
)

SET(LIBS cc3d::Field3D cc3d::Logger)

ADD_SHARED_LIBRARY(Boundary ${SRCS} LINK_LIBRARIES ${LIBS})

TARGET_INCLUDE_DIRECTORIES(${BoundaryLib} PUBLIC
   $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D/Boundary .h
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
