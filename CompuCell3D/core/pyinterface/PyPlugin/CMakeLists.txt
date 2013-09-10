SET(SRCS
   PyPlugin.cpp
   PyCompuCellObjAdapter.cpp
   EnergyFunctionPyWrapper.cpp
   ChangeWatcherPyWrapper.cpp
   StepperPyWrapper.cpp
   TypeChangeWatcherPyWrapper.cpp
   NeighborFinderParams.cpp
   PyAttributeAdder.cpp
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_LIBRARY(PyPlugin STATIC ${SRCS})
  target_link_libraries(PyPlugin Field3DStatic Potts3DStatic CompuCellLibShared ${PYTHON_LIBRARIES})


else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_LIBRARY(PyPlugin ${SRCS})
  target_link_libraries(PyPlugin Field3DShared CompuCellLibShared ${PYTHON_LIBRARIES})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

IF(USE_LIBRARY_VERSIONS)
    SET_TARGET_PROPERTIES(
    PyPlugin PROPERTIES
    ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)


INSTALL_TARGETS(/lib
  PyPlugin)

INSTALL_FILES(/include/pyinterface/PyPlugin .h
   PyPlugin 
   PyCompuCellObjAdapter 
   EnergyFunctionPyWrapper 
   ChangeWatcherPyWrapper 
   StepperPyWrapper 
   TypeChangeWatcherPyWrapper 
   NeighborFinderParams 
   PyAttributeAdder
)

# INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})
