SET(SRCS
   FiPyInterface.cpp
)


INCLUDE_DIRECTORIES(
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/simthread
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/src
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${COMPUCELL3D_SOURCE_DIR}/core
   ${COMPUCELL3D_SOURCE_DIR}/core/Utils
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
   ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin

    )
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsStatic
       ${PYTHON_LIBRARIES}
    )

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared
       ${PYTHON_LIBRARIES}
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FiPyInterface SHARED ${SRCS})
    
    target_link_libraries(FiPyInterface  ${LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FiPyInterface ${SRCS})

    target_link_libraries(FiPyInterface  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  FiPyInterface)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FiPyInterface
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FiPyInterface
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/FiPyInterface
   ${PYTHON_INCLUDE_PATH}
)

SET_SOURCE_FILES_PROPERTIES(FiPyInterface.i PROPERTIES CPLUSPLUS ON)
# SET_SOURCE_FILES_PROPERTIES(FiPyInterface.i PROPERTIES SWIG_FLAGS "-includeall") #use on fewer files!

SWIG_ADD_MODULE(FiPyInterface python FiPyInterface.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_FiPyInterface PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET_TARGET_PROPERTIES(_FiPyInterface PROPERTIES OUTPUT_NAME  "_FiPyInterface")   
    
SWIG_LINK_LIBRARIES(FiPyInterface ${LIBS_SWIG})

install_targets(/lib/python _FiPyInterface)
set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/FiPyInterface)

INSTALL(FILES ${python_files_path}/FiPyInterface.py
	      DESTINATION lib/python)
