# Recurse into subprojects

SET(SRCS
  SerializerDE.cpp
  )



  
# FIND_PACKAGE(VTK REQUIRED)
# message("VTK_MAJOR_VERSION=" ${VTK_MAJOR_VERSION})

if (${VTK_MAJOR_VERSION} EQUAL "6")    
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "5")    
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()


INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${VTK_INCLUDE_DIRS}
   ${COMPUCELL3D_SOURCE_DIR}/core
   ${COMPUCELL3D_SOURCE_DIR}/core/Utils
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins

    )

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsStatic
       FieldExtractor
       ${Python_LIBRARY_RELEASE}
       ${VTK_LIBS}
       # simthreadShared

    )

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared
       FieldExtractor
       ${PYTHON_LIBRARY_TO_LINK}
       ${VTK_LIBS}
       # simthreadShared

    )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    
    
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared       
       FieldExtractor
       ${Python_LIBRARY_RELEASE}
       ${VTK_LIBS} 
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  
INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core/SerializerDE
    )

    
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE SHARED ${SRCS})
    
    target_link_libraries(SerializerDE  ${LIBS})

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
   ADD_LIBRARY(SerializerDE ${SRCS})

    target_link_libraries(SerializerDE  ${LIBS})
    set_target_properties(SerializerDE PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")


else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE ${SRCS})

    target_link_libraries(SerializerDE  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib/site-packages/cc3d/cpp/lib RUNTIME_DIRECTORY /lib/site-packages/cc3d/cpp/bin SerializerDE)
        
INSTALL_FILES(/include/SerializerDE .h
SerializerDE
)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   SerializerDE
   ${LIBS}
   ${Python_LIBRARY_RELEASE}

)

elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    SET(LIBS_SWIG
        SerializerDE
        ${LIBS}
        ${PYTHON_LIBRARY_TO_LINK}

    )


else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   SerializerDE
   ${LIBS}
   ${Python_LIBRARY_RELEASE}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/SerializerDE
   ${PYTHON_INCLUDE_PATH}
   ${VTK_INCLUDE_DIRS}
)



SET_SOURCE_FILES_PROPERTIES(SerializerDEPy.i PROPERTIES CPLUSPLUS ON)

# SWIG_ADD_MODULE(SerializerDEPy python SerializerDEPy.i)

SWIG_ADD_LIBRARY(SerializerDEPy LANGUAGE python SOURCES SerializerDEPy.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    
SWIG_LINK_LIBRARIES(SerializerDEPy ${LIBS_SWIG})

set_target_properties(SerializerDEPy PROPERTIES LINK_FLAGS ${undefined_link_flags})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for SerializerDEPy.so
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which SerializerDEPy.so depends on are
   # installed relative to where SerializerDEPy.py is installed
   SET_TARGET_PROPERTIES(SerializerDEPy PROPERTIES INSTALL_RPATH "@loader_path/lib")


endif()



# install_targets(/lib/python _SerializerDEPy)

install(TARGETS SerializerDEPy DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/SerializerDE)

INSTALL(FILES ${python_files_path}/SerializerDEPy.py
	      DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})

# INSTALL(FILES ${python_files_path}/SerializerDEPy.py
# 	      DESTINATION lib/python)


#INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})
