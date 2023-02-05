SET(SRCS
  AdjacentNeighbor.cpp
  Field3DChangeWatcher.cpp
  Field3DIndex.cpp
  Point3D.cpp
  Dim3D.cpp
  Field3D.cpp
  NeighborFinder.cpp
  WatchableField3D.cpp
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES cc3d::Logger)
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES cc3d::Logger)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

TARGET_INCLUDE_DIRECTORIES(${Field3DLib} PUBLIC
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
  ${Python3_INCLUDE_DIRS}
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D/Field3D .h
  AdjacentNeighbor
  Dim3D
  Array3D
  Field3DChangeWatcher
  Field3D
  Field3DImpl
  Field3DIndex
  Field3DIO
  NeighborFinder
  Neighbor
  Point3D
  WatchableField3D
  CC3D_Field3D
)
