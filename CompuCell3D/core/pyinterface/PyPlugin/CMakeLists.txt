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
  target_link_libraries(PyPlugin Field3DStatic Potts3DStatic CompuCellLibShared ${Python_LIBRARY_RELEASE})

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

  ADD_LIBRARY(PyPlugin ${SRCS})
  target_link_libraries(PyPlugin Field3DShared CompuCellLibShared ${PYTHON_LIBRARY_TO_LINK})
  set_target_properties(PyPlugin PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")


else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_LIBRARY(PyPlugin ${SRCS})
  target_link_libraries(PyPlugin Field3DShared CompuCellLibShared ${Python_LIBRARY_RELEASE})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

IF(USE_LIBRARY_VERSIONS)
    SET_TARGET_PROPERTIES(
    PyPlugin PROPERTIES
    ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)


INSTALL_TARGETS(
    "/${COMPUCELL3D_INSTALL_LIB_DIR}"
    # /lib
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
