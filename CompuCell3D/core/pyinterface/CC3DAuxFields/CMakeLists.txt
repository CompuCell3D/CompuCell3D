SET(LIBS
    cc3d::CompuCellLib
    ${PYTHON_LIBRARY_TO_LINK}
)

#set(HEADER_FILES_USED_BY_SWIG NumpyArrayWrapper.h)


SET_SOURCE_FILES_PROPERTIES(CC3DAuxFields.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(CC3DAuxFields LANGUAGE python SOURCES CC3DAuxFields.i)



## Ensure SWIG reruns when the header files change
#add_dependencies(CC3DAuxFields ${HEADER_FILES_USED_BY_SWIG})


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(CC3DAuxFields PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

target_include_directories(CC3DAuxFields PUBLIC
  ${PYTHON_DIRS_TO_INCLUDE}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/Field3D>
  $<TARGET_PROPERTY:cc3d::XMLUtils,INCLUDE_DIRECTORIES>
)
set_target_properties(CC3DAuxFields PROPERTIES
  LINK_FLAGS ${undefined_link_flags}
  SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
)
SWIG_LINK_LIBRARIES(CC3DAuxFields ${LIBS})
target_link_directories(CC3DAuxFields PUBLIC ${Python_LIBRARY_DIRS})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for CC3DAuxFields.so
   SET_TARGET_PROPERTIES(CC3DAuxFields PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(CC3DAuxFields PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which CC3DAuxFields.so depends on are
   # installed relative to where CC3DAuxFields.py is installed
   SET_TARGET_PROPERTIES(CC3DAuxFields PROPERTIES INSTALL_RPATH "@loader_path/lib;@loader_path/../../../")


endif()

install(TARGETS CC3DAuxFields DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} )

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/CC3DAuxFields)

INSTALL(FILES 
    ${python_files_path}/CC3DAuxFields.py
	DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
)

