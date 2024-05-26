SET(SRCS
    SerializerDE.cpp
)

if (${VTK_MAJOR_VERSION} EQUAL "6")    
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "5")    
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()

set(INCLUDE_DIRS
    ${VTK_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins>
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D/plugins>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/pyinterface/PyPlugin>
)

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})

SET(LIBS
    cc3d::CompuCellLib
    FieldExtractor
    ${PYTHON_LIBRARY_TO_LINK}
    ${VTK_LIBS}

)
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE SHARED ${SRCS})

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    ADD_LIBRARY(SerializerDE ${SRCS})

    set_target_properties(SerializerDE PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE ${SRCS})

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

target_link_libraries(SerializerDE ${LIBS})
target_include_directories(SerializerDE PUBLIC ${INCLUDE_DIRS})

install(TARGETS SerializerDE
        EXPORT SerializerDE
        RUNTIME DESTINATION ${COMPUCELL3D_RUNTIME_DESTINATION}
        LIBRARY DESTINATION ${COMPUCELL3D_LIBRARY_DESTINATION}
        ARCHIVE DESTINATION ${COMPUCELL3D_ARCHIVE_DESTINATION}

#        RUNTIME DESTINATION ${COMPUCELL3D_INSTALL_BIN_DIR}
#        LIBRARY DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
#        ARCHIVE DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
)
EXPORT_COMPONENT(SerializerDE SerializerDE)

INSTALL_FILES(/include/SerializerDE .h
    SerializerDE
)

SET(LIBS_SWIG
    SerializerDE
    ${LIBS}
    ${PYTHON_LIBRARY_TO_LINK}

)

SET_SOURCE_FILES_PROPERTIES(SerializerDEPy.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(SerializerDEPy LANGUAGE python SOURCES SerializerDEPy.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(SerializerDEPy ${LIBS_SWIG})
target_link_directories(SerializerDEPy PUBLIC ${Python3_LIBRARY_DIRS})

set_target_properties(SerializerDEPy PROPERTIES LINK_FLAGS ${undefined_link_flags})
# set_property(TARGET SerializerDEPy PROPERTY SWIG_INCLUDE_DIRECTORIES
#     # ${PYTHON_INCLUDE_PATH}
#     ${PYTHON_DIRS_TO_INCLUDE}
#     ${VTK_INCLUDE_DIRS}
#     $<TARGET_PROPERTY:SerializerDE,INCLUDE_DIRECTORIES>
# )

target_include_directories(SerializerDEPy PUBLIC
    ${PYTHON_DIRS_TO_INCLUDE}
    ${VTK_INCLUDE_DIRS}
    $<TARGET_PROPERTY:SerializerDE,INCLUDE_DIRECTORIES>
)

#message("ppy include path ",  ${PYTHON_INCLUDE_PATH})

#message("ppy include path ",  ${PYTHON_DIRS_TO_INCLUDE})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for SerializerDEPy.so
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which SerializerDEPy.so depends on are
   # installed relative to where SerializerDEPy.py is installed
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES INSTALL_RPATH "@loader_path/lib;@loader_path/../../../")


endif()

install(TARGETS SerializerDEPy DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/SerializerDE)

INSTALL(FILES ${python_files_path}/SerializerDEPy.py
        DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
)
