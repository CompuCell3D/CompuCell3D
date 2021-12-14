SET(LIBS
    cc3d::XMLUtils
    ${PYTHON_LIBRARY_TO_LINK}
)

SET_SOURCE_FILES_PROPERTIES(CC3DXML.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(CC3DXML LANGUAGE python SOURCES CC3DXML.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(CC3DXML PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

target_include_directories(CC3DXML PUBLIC
  ${PYTHON_DIRS_TO_INCLUDE}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/XMLUtils>
  $<TARGET_PROPERTY:cc3d::XMLUtils,INCLUDE_DIRECTORIES>
)
set_target_properties(CC3DXML PROPERTIES
  LINK_FLAGS ${undefined_link_flags}
  SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
)
SWIG_LINK_LIBRARIES(CC3DXML ${LIBS})
target_link_directories(CC3DXML PUBLIC ${Python_LIBRARY_DIRS})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for CC3DXML.so
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which CC3DXML.so depends on are
   # installed relative to where CC3DXML.py is installed
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES INSTALL_RPATH "@loader_path/lib;@loader_path/../../../")


endif()

install(TARGETS CC3DXML DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} )

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/CC3DXML)

INSTALL(FILES 
    ${python_files_path}/CC3DXML.py
	DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
)

