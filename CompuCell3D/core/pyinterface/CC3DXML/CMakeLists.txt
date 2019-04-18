

# set(PYTHON_LIBRARY_TO_LINK ${PYTHON_LIBRARIES})
# set(undefined_link_flags )
# if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
#     set(PYTHON_LIBRARY_TO_LINK )
#     set(undefined_link_flags "-undefined dynamic_lookup")
# endif()

SET(LIBS
   XMLUtilsShared
   ${PYTHON_LIBRARY_TO_LINK}
)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
    ${COMPUCELL3D_SOURCE_DIR}/core/XMLUtils
    # ${COMPUCELL3D_SOURCE_DIR}/BasicUtils
)

SET_SOURCE_FILES_PROPERTIES(CC3DXML.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_LIBRARY(CC3DXML LANGUAGE python SOURCES CC3DXML.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(CC3DXML PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(CC3DXML ${LIBS})
set_target_properties(CC3DXML PROPERTIES LINK_FLAGS ${undefined_link_flags})

if (APPLE)

   # interesting reading: https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/

   # INSTALL_NAME_DIR overrides MACOSX_RPATH.  Try not to do that.

   # Use rpath for CC3DXML.so
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which CC3DXML.so depends on are
   # installed relative to where CC3DXML.py is installed
   SET_TARGET_PROPERTIES(CC3DXML PROPERTIES INSTALL_RPATH "@loader_path/lib")


endif()


# install(TARGETS _CC3DXML    DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} )
install(TARGETS CC3DXML    DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} )

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/CC3DXML)

INSTALL(FILES 
    ${python_files_path}/CC3DXML.py
	DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
)

