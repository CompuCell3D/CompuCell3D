MESSAGE(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})

SET(SRCS
   FieldStorage.cpp
   FieldExtractorBase.cpp
   FieldExtractor.cpp
   FieldExtractorCML.cpp
   FieldWriter.cpp
   FieldWriterCML.cpp
   FieldStreamer.cpp
)

if (${VTK_MAJOR_VERSION} EQUAL "6")
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "7")
    SET(VTK_LIBS vtkRenderingVolume vtkInteractionStyle  vtkIOLegacy)
endif()


if (${VTK_MAJOR_VERSION} EQUAL "8")
    SET(VTK_LIBS vtkRenderingVolume vtkInteractionStyle  vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "9")
#    SET(VTK_LIBS vtkRenderingVolume vtkInteractionStyle  vtkIOLegacy)
    find_package(VTK COMPONENTS
            RenderingVolume InteractionStyle  IOLegacy
            )
    #include(${VTK_USE_FILE})
    SET(VTK_LIBS ${VTK_LIBRARIES})
endif()

if (${VTK_MAJOR_VERSION} EQUAL "5")
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})

SET(LIBS
    cc3d::CompuCellLib
    cc3d::NeighborTracker
    cc3d::FocalPointPlasticity
    cc3d::PublicUtilities
    ${PYTHON_LIBRARY_TO_LINK}
    ${VTK_LIBS}

)

MESSAGE("VTK_LIB_DIRS " ${VTK_LIBRARY_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor SHARED ${SRCS})

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    ADD_LIBRARY(FieldExtractor ${SRCS})
    set_target_properties(FieldExtractor PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(FieldExtractor ${SRCS})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

TARGET_INCLUDE_DIRECTORIES(FieldExtractor PUBLIC
    ${VTK_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/pyinterface>
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin>
)
target_link_libraries(FieldExtractor ${LIBS})

install(TARGETS FieldExtractor
        EXPORT FieldExtractor
        RUNTIME DESTINATION ${COMPUCELL3D_RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${COMPUCELL3D_LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${COMPUCELL3D_ARCHIVE_DESTINATION}

#        RUNTIME DESTINATION ${COMPUCELL3D_INSTALL_BIN_DIR}
#        LIBRARY DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
#        ARCHIVE DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
)
EXPORT_COMPONENT(FieldExtractor FieldExtractor)

SET(LIBS_SWIG
    FieldExtractor
    ${LIBS}
    ${PYTHON_LIBRARY_TO_LINK}

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

SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES
    OUTPUT_NAME "PlayerPython"
    LINK_FLAGS ${undefined_link_flags}
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
)

target_include_directories(PlayerPythonNew PUBLIC
    ${PYTHON_DIRS_TO_INCLUDE}
    ${VTK_INCLUDE_DIRS}
    $<TARGET_PROPERTY:FieldExtractor,INCLUDE_DIRECTORIES>
)

SWIG_LINK_LIBRARIES(PlayerPythonNew ${LIBS_SWIG})
target_link_directories(PlayerPythonNew PUBLIC ${Python3_LIBRARY_DIRS})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for PlayerPythonNew.so
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which PlayerPythonNew.so depends on are
   # installed relative to where PlayerPythonNew.py is installed
   SET_TARGET_PROPERTIES(PlayerPythonNew PROPERTIES INSTALL_RPATH "@loader_path/lib;@loader_path/../../../")


endif()

install(TARGETS PlayerPythonNew DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/PlayerPythonNew)

INSTALL(FILES ${python_files_path}/PlayerPython.py
	    DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
)
