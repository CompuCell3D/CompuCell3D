
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

FIND_PACKAGE(PythonLibs)

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core ${PYTHON_INCLUDE_PATH} )


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Field3D ${SRCS}  LINK_LIBRARIES BasicUtilsStatic ${PYTHON_LIBRARIES})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES BasicUtilsShared ${PYTHON_LIBRARIES})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


INSTALL_FILES(/include/CompuCell3D/CompuCell3D/Field3D .h
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
