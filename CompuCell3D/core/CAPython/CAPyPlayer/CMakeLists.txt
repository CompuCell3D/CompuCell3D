

# SET(NUMPY_INCLUDE_DIR )
# EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/PlayerPythonNew ARGS NumpyHeader.py
      # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

#   
MESSAGE(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})

SET(SRCS
   # # # CAFieldStorage.cpp
   # # # FieldExtractorBase.cpp   
   CAFieldExtractor.cpp
   # # # FieldExtractorCML.cpp
   # # # FieldWriter.cpp
)
FIND_PACKAGE(VTK REQUIRED)
if (${VTK_MAJOR_VERSION} EQUAL "6")    
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "5")    
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()

INCLUDE_DIRECTORIES(
   # # # ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/simthread
   # # # ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/src
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${VTK_INCLUDE_DIRS}
   ${COMPUCELL3D_SOURCE_DIR}/core
   ${COMPUCELL3D_SOURCE_DIR}/core/Utils
   # # # ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
   # # # ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
   # # # ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin
   # ${NUMPY_INCLUDE_DIR}
   # ${COMPUCELL3D_SOURCE_DIR}/BasicUtils
   # ${PYINTERFACE_SRC_DIR}/PyPlugin
    )

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CAShared
       BasicUtilsStatic
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS}	
       # simthreadShared

    )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CAShared
       BasicUtilsShared
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS}
       
       # simthreadShared
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

MESSAGE("VTK_LIB_DIRS " ${VTK_LIBRARY_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_SHARED_LIBRARY(CAFieldUtils ${SRCS} LINK_LIBRARIES ${LIBS})
    # ADD_LIBRARY(CAFieldUtils SHARED ${SRCS})
    
    # target_link_libraries(CAFieldUtils  ${LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_SHARED_LIBRARY(CAFieldUtils ${SRCS} LINK_LIBRARIES ${LIBS})
    # ADD_LIBRARY(CAFieldUtils ${SRCS})

    # target_link_libraries(CAFieldUtils  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  CAFieldUtilsShared)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   CAFieldUtilsShared
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   CAFieldUtilsShared
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   # # # ${PYINTERFACE_SRC_DIR}/PlayerPythonNew
   ${PYTHON_INCLUDE_PATH}
   ${VTK_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(CAFieldUtils.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(CAFieldUtils python CAFieldUtils.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_CAFieldUtils PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET_TARGET_PROPERTIES(_CAFieldUtils PROPERTIES OUTPUT_NAME  "_CAFieldUtils")   
    
SWIG_LINK_LIBRARIES(CAFieldUtils ${LIBS_SWIG})

install_targets(/lib/python _CAFieldUtils)
set(python_files_path ${CMAKE_BINARY_DIR}/core/CAPython/CAPyPlayer)

INSTALL(FILES ${python_files_path}/CAFieldUtils.py
	      DESTINATION lib/python)

