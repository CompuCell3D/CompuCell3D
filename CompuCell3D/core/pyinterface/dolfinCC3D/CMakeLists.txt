# SET(EXTRA_LIBS )
# if(USE_DOLFIN)
#   add_definitions(-DCC3D_HAS_DOLFIN)
#   message("CC3DHAVEDOLFIN")
#   SET(EXTRA_LIBS dolfin)
# endif(USE_DOLFIN)

# Get DOLFIN configuration data (dolfin-config.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(dolfin)  # /usr/share/dolfin/cmake/dolfin-config.cmake - this file contains dolfin configurationvariables
# message("DOLFIN INCLUDE " ${DOLFIN_INCLUDE_DIRS})
# message("DOLFIN_3RD_PARTY_INCLUDE_DIRS " ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

FIND_PACKAGE(Boost)
message("THIS IS BOOST VAR" ${Boost_INCLUDE_DIRS})

SET(SRCS
   dolfinCC3D.cpp
   CleaverDolfinUtil.cpp
   CellFieldCleaverSimulator.cpp
   CustomSubDomains.cpp
   CustomExpressions.cpp
)



SET(EXTRA_LIBS cleaver dolfin)

ADD_DEFINITIONS(-I/usr/include -I/usr/include/vtk-5.8 -I/usr/include/qt4 -I/usr/lib/openmpi/include/openmpi -I/usr/lib/openmpi/include -I/usr/include/suitesparse -I/usr/include/spooles -I/usr/include/scotch -I/usr/lib/petscdir/3.2/linux-gnu-c-opt/include -I/usr/lib/petscdir/3.2/include -I/usr/lib/slepcdir/3.2/include -I/usr/lib/slepcdir/3.2/linux-gnu-c-opt/include -I/usr/lib/slepcdir/3.2 -I/usr/include/libxml2 -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/include -I/usr/include/python2.7 -DMPICH_IGNORE_CXX_SEEK -Wno-deprecated -g -O2 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Wformat-security -Werror=format-security -D_FORTIFY_SOURCE=2 -frounding-math -fopenmp -DDOLFIN_VERSION=\"1.1.0\" -DBOOST_UBLAS_NDEBUG -DHAS_HDF5 -DHAS_SLEPC -DHAS_PETSC -DHAS_UMFPACK -DHAS_CHOLMOD -DHAS_SCOTCH -DHAS_CGAL -DHAS_ZLIB -DHAS_MPI -DHAS_OPENMP -DHAS_QT4 -DHAS_VTK -DHAS_QVTK)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(dolfinUtils ${SRCS})
    target_link_libraries(dolfinCC3D  ${Python3_LIBRARIES} ${EXTRA_LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(dolfinUtils ${SRCS})
    target_link_libraries(dolfinUtils  ${Python3_LIBRARIES} ${EXTRA_LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  dolfinUtils)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   dolfinUtils
   ${Python3_LIBRARIES}
   ${EXTRA_LIBS}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   dolfinUtils
   ${Python3_LIBRARIES}
   ${EXTRA_LIBS}
)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/dolfinCC3D
   ${Python3_INCLUDE_PATH}
   ${DOLFIN_INCLUDE_DIRS}
#    ${Boost_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(dolfinCC3D.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(dolfinCC3D python dolfinCC3D.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_dolfinCC3D PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(dolfinCC3D ${LIBS})
target_link_directories(dolfinCC3D PUBLIC ${Python3_LIBRARY_DIRS})

install(TARGETS _dolfinCC3D    DESTINATION lib/python )



set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/dolfinCC3D)

INSTALL(FILES ${python_files_path}/dolfinCC3D.py
	      DESTINATION lib/python)

##############################################################################################33

# SET_SOURCE_FILES_PROPERTIES(dolfinCC3DExpr.i PROPERTIES CPLUSPLUS ON)

# SWIG_ADD_MODULE(dolfinCC3DExpr python dolfinCC3DExpr.i)

# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#   SET_TARGET_PROPERTIES(_dolfinCC3DExpr PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# SWIG_LINK_LIBRARIES(dolfinCC3DExpr ${LIBS})


# install(TARGETS _dolfinCC3DExpr    DESTINATION lib/python )



# set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/dolfinCC3D)

# INSTALL(FILES ${python_files_path}/dolfinCC3DExpr.py
# 	      DESTINATION lib/python)

          
# #SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadStatic)
# #SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadShared)
