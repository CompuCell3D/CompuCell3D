

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


if (${VTK_MAJOR_VERSION} EQUAL "8")
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
       ${Python_LIBRARY_RELEASE}
       ${VTK_LIBS}
       # simthreadShared

    )
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared
       ${PYTHON_LIBRARY_TO_LINK}
       ${VTK_LIBS}

       # simthreadShared
    )

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared
       ${Python_LIBRARY_RELEASE}
       ${VTK_LIBS}

       # simthreadShared
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

MESSAGE("VTK_LIB_DIRS " ${VTK_LIBRARY_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor SHARED ${SRCS})

    target_link_libraries(FieldExtractor  ${LIBS})

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
   ADD_LIBRARY(FieldExtractor ${SRCS})

    target_link_libraries(FieldExtractor  ${LIBS})
    set_target_properties(FieldExtractor PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor ${SRCS})

    target_link_libraries(FieldExtractor  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib/site-packages/cc3d/cpp/lib RUNTIME_DIRECTORY /lib/site-packages/cc3d/cpp/bin FieldExtractor)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FieldExtractor
   ${LIBS}
   ${Python_LIBRARY_RELEASE}

)

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    SET(LIBS_SWIG
        FieldExtractor
        ${LIBS}
        ${PYTHON_LIBRARY_TO_LINK}

    )


else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   FieldExtractor
   ${LIBS}
   ${Python_LIBRARY_RELEASE}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/PlayerPythonNew
   ${VTK_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(PlayerPython.i PROPERTIES CPLUSPLUS ON)

# have to add additional flag here for Windows because _WIN32 is unreliable in swig. it does not work properly with typedefs
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET_SOURCE_FILES_PROPERTIES(PlayerPython.i PROPERTIES SWIG_FLAGS "-DSWIGWIN")
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


SWIG_ADD_LIBRARY(PlayerPythonNew LANGUAGE python SOURCES PlayerPython.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES OUTPUT_NAME  "PlayerPython")

SWIG_LINK_LIBRARIES(PlayerPythonNew ${LIBS_SWIG})
set_target_properties(PlayerPythonNew PROPERTIES LINK_FLAGS ${undefined_link_flags})
  
if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for PlayerPythonNew.so
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which PlayerPythonNew.so depends on are
   # installed relative to where PlayerPythonNew.py is installed
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES INSTALL_RPATH "@loader_path/lib")


endif()
  
  

# install_targets(${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} _PlayerPythonNew)
install(TARGETS PlayerPythonNew DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/PlayerPythonNew)

INSTALL(FILES ${python_files_path}/PlayerPython.py
	      DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadStatic)
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadShared)
