

# SET(NUMPY_INCLUDE_DIR )
# EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/PlayerPythonNew ARGS NumpyHeader.py
      # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

#
MESSAGE(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})

SET(SRCS
   FieldStorage.cpp
   FieldExtractorBase.cpp
   FieldExtractor.cpp
   FieldExtractorCML.cpp
   FieldWriter.cpp
)
FIND_PACKAGE(VTK REQUIRED)
if (${VTK_MAJOR_VERSION} EQUAL "6")
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "7")
    SET(VTK_LIBS vtkRenderingVolume vtkInteractionStyle  vtkIOLegacy)
endif()


if (${VTK_MAJOR_VERSION} EQUAL "5")
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()

INCLUDE_DIRECTORIES(
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/simthread
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCellPlayer/src
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${VTK_INCLUDE_DIRS}
   ${COMPUCELL3D_SOURCE_DIR}/core
   ${COMPUCELL3D_SOURCE_DIR}/core/Utils
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
   ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin
   # ${NUMPY_INCLUDE_DIR}
   # ${COMPUCELL3D_SOURCE_DIR}/BasicUtils
   # ${PYINTERFACE_SRC_DIR}/PyPlugin
    )

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsStatic
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS}
       # simthreadShared

    )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS}

       # simthreadShared
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

MESSAGE("VTK_LIB_DIRS " ${VTK_LIBRARY_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor SHARED ${SRCS})

    target_link_libraries(FieldExtractor  ${LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor ${SRCS})

    target_link_libraries(FieldExtractor  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  FieldExtractor)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FieldExtractor
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FieldExtractor
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/PlayerPythonNew
   ${PYTHON_INCLUDE_PATH}
   ${VTK_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(PlayerPython.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(PlayerPythonNew python PlayerPython.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_PlayerPythonNew PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET_TARGET_PROPERTIES(_PlayerPythonNew PROPERTIES OUTPUT_NAME  "_PlayerPython")

SWIG_LINK_LIBRARIES(PlayerPythonNew ${LIBS_SWIG})

install_targets(/lib/python _PlayerPythonNew)
set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/PlayerPythonNew)

INSTALL(FILES ${python_files_path}/PlayerPython.py
	      DESTINATION lib/python)

#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadStatic)
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadShared)
