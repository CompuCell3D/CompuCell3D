ADD_CA_MODULE(CanonicalProbability LINK_LIBRARIES ${MODULE_LIBS} )

# SET(MODULE_NAME CanonicalProbability)

# # MESSAGE(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR})

# SET(SRCS
   # CanonicalProbability.cpp
# )


# INCLUDE_DIRECTORIES(
   # ${CMAKE_CURRENT_SOURCE_DIR}
   # ${COMPUCELL3D_SOURCE_DIR}/core
   # ${COMPUCELL3D_SOURCE_DIR}/core/Utils
    # )

# # LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})
    
# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # SET(LIBS
       # CAShared
       # BasicUtilsStatic
       # ${PYTHON_LIBRARIES}
    # )
# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # SET(LIBS
       # CAShared
       # BasicUtilsShared
       # ${PYTHON_LIBRARIES}
    # )
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # ADD_SHARED_LIBRARY(${MODULE_NAME} ${SRCS} LINK_LIBRARIES ${LIBS})
# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # ADD_SHARED_LIBRARY(${MODULE_NAME} ${SRCS} LINK_LIBRARIES ${LIBS})
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# INSTALL_TARGETS(/lib
  # ${MODULE_NAME}Shared)




# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# SET(LIBS_SWIG
   # ${MODULE_NAME}Shared
   # ${LIBS}
   # ${PYTHON_LIBRARIES}

# )

# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# SET(LIBS_SWIG
   # ${MODULE_NAME}Shared
   # ${LIBS}
   # ${PYTHON_LIBRARIES}

# )

# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# INCLUDE_DIRECTORIES(
   # ${CMAKE_CURRENT_SOURCE_DIR}
   # # # # ${PYINTERFACE_SRC_DIR}/PlayerPythonNew
   # ${PYTHON_INCLUDE_PATH}
# )

# SET_SOURCE_FILES_PROPERTIES(${MODULE_NAME}.i PROPERTIES CPLUSPLUS ON)

# SWIG_ADD_MODULE(${MODULE_NAME} python ${MODULE_NAME}.i)

# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  # SET_TARGET_PROPERTIES(_${MODULE_NAME} PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# SET_TARGET_PROPERTIES(_${MODULE_NAME} PROPERTIES OUTPUT_NAME  "_${MODULE_NAME}")   
    
# SWIG_LINK_LIBRARIES(${MODULE_NAME} ${LIBS_SWIG})

# install_targets(/lib/python/CA _${MODULE_NAME})
# set(python_files_path ${CMAKE_BINARY_DIR}/core/CA/modules/${MODULE_NAME})

# INSTALL(FILES ${python_files_path}/${MODULE_NAME}.py
	      # DESTINATION lib/python/CA)

# #folder setup for better display iof projects in the IDE's          
# SET_PROPERTY(TARGET ${MODULE_NAME}Shared  PROPERTY FOLDER "CA/${MODULE_NAME}/${MODULE_NAME}")
# SET_PROPERTY(TARGET _${MODULE_NAME}  PROPERTY FOLDER "CA/${MODULE_NAME}/${MODULE_NAME}")
          
