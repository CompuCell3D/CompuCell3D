
SET(SRCS
  CellCM.cpp
  SimulationBox.cpp
  CellInventoryCM.cpp
  CellFactoryCM.cpp
  )

FIND_PACKAGE(PythonLibs)

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core  ${COMPUCELL3D_SOURCE_DIR}/CenterModel ${PYTHON_INCLUDE_PATH} )


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Components ${SRCS}  LINK_LIBRARIES BasicUtilsStatic  PublicUtilitiesStatic BoundaryShared ${PYTHON_LIBRARIES})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Field3D ${SRCS} LINK_LIBRARIES BasicUtilsShared ${PYTHON_LIBRARIES})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


INSTALL_FILES(/include/CompuCell3D/CenterModel/Components .h
  CellCM
  SimulationBox
  CellInventoryCM
  CellFactoryCM
  
  )
