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

SET(LIBS 
  cc3d::Field3D 
  cc3d::CompuCellLib 
  ${PYTHON_LIBRARY_TO_LINK}
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_LIBRARY(PyPlugin STATIC ${SRCS})
  LIST(APPEND LIBS cc3d::Potts3D)

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

  ADD_LIBRARY(PyPlugin ${SRCS})
  set_target_properties(PyPlugin PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_LIBRARY(PyPlugin ${SRCS})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

target_link_libraries(PyPlugin ${LIBS})
target_include_directories(PyPlugin PUBLIC 
  ${PYTHON_DIRS_TO_INCLUDE} 
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core> 
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
)

IF(USE_LIBRARY_VERSIONS)
    SET_TARGET_PROPERTIES(
      PyPlugin PROPERTIES
      ${COMPUCELL3D_LIBRARY_PROPERTIES}
    )
ENDIF(USE_LIBRARY_VERSIONS)

install(TARGETS PyPlugin 
        EXPORT PyPlugin
        RUNTIME DESTINATION ${COMPUCELL3D_RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${COMPUCELL3D_LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${COMPUCELL3D_ARCHIVE_DESTINATION}

#        RUNTIME DESTINATION ${COMPUCELL3D_INSTALL_BIN_DIR}
#        LIBRARY DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
#        ARCHIVE DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
)
EXPORT_COMPONENT(PyPlugin PyPlugin)

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
