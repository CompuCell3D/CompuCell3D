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
)

SET(EXTRA_LIBS cleaver dolfin)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(dolfinUtils ${SRCS})
    target_link_libraries(dolfinCC3D  ${PYTHON_LIBRARIES} ${EXTRA_LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(dolfinUtils ${SRCS})
    target_link_libraries(dolfinUtils  ${PYTHON_LIBRARIES} ${EXTRA_LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  dolfinUtils)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   dolfinUtils
   ${PYTHON_LIBRARIES}
   ${EXTRA_LIBS}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   dolfinUtils
   ${PYTHON_LIBRARIES}
   ${EXTRA_LIBS}
)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/dolfinCC3D
   ${PYTHON_INCLUDE_PATH}
   ${DOLFIN_INCLUDE_DIRS}
#    ${Boost_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(dolfinCC3D.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(dolfinCC3D python dolfinCC3D.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_dolfinCC3D PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(dolfinCC3D ${LIBS})


install(TARGETS _dolfinCC3D    DESTINATION lib/python )



set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/dolfinCC3D)

INSTALL(FILES ${python_files_path}/dolfinCC3D.py
	      DESTINATION lib/python)

          
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadStatic)
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadShared)
