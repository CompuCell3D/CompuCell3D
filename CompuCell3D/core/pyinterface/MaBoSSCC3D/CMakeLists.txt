# If on windows and not building, install pre-builts and return
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND NOT MINGW)
    install(
        FILES 
        ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS/engine/win/MaBoSSCC3DPy.py 
        ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS/engine/win/_MaBoSSCC3DPy.pyd 
        DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
    )
    return()
endif()

set_source_files_properties(MaBoSSCC3D.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_LIBRARY(MaBoSSCC3DPy LANGUAGE python SOURCES MaBoSSCC3D.i)

target_include_directories(MaBoSSCC3DPy PUBLIC 
    ${Python_INCLUDE_DIRS} 
    ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS 
    ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS/engine/src
)
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    target_include_directories(MaBoSSCC3DPy PUBLIC ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS/engine/extern)
endif()

set_target_properties(MaBoSSCC3DPy PROPERTIES 
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE 
    LINK_FLAGS ${undefined_link_flags}
)
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # set dll name to pyd for compatibility with Python standard
    set_target_properties(MaBoSSCC3DPy PROPERTIES SUFFIX ".pyd")
endif()

SWIG_LINK_LIBRARIES(MaBoSSCC3DPy 
    MaBoSSCC3DShared 
    ${PYTHON_LIBRARY_TO_LINK}
)

if(APPLE)

   # Use rpath for CC3DMaBoSS.so
   set_target_properties(MaBoSSCC3DPy PROPERTIES MACOSX_RPATH TRUE)

   # append directories in the linker search path and outside the project to the INSTALL_RPATH
   set_target_properties(MaBoSSCC3DPy PROPERTIES CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # this is where libraries on which CC3DMaBoSS.so depends on are
   # installed relative to where CC3DMaBoSS.py is installed
   set_target_properties(MaBoSSCC3DPy PROPERTIES INSTALL_RPATH "@loader_path/lib")

endif()

install(TARGETS MaBoSSCC3DPy DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})
install(FILES ${CMAKE_BINARY_DIR}/core/pyinterface/MaBoSSCC3D/MaBoSSCC3DPy.py
        DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})
