
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    set(config_header_name )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
        set(dynamic_lib_extension so)
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

    if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
        set(dynamic_lib_extension dylib)
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

    
    configure_file(${PROJECT_SOURCE_DIR}/include/sbmlsolver/config.h.in ${PROJECT_SOURCE_DIR}/include/sbmlsolver/config.h)
    set(config_header_name config.h)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

Set(Sources
    odeConstruct.c
    nullSolver.c
    modelSimplify.c
    interpol.c
    integratorSettings.c
    integratorInstance.c
    drawGraph.c
    daeSolver.c
    cvodeSolver.c
    cvodeData.c
    compiler.c
    charBuffer.cpp
    ASTIndexNameNode.cpp
    arithmeticCompiler.c
    util.c
    solverError.c
    sensSolver.c
    sbmlResults.c
    sbml.c
    processAST.c
    odeSolver.c
    odeModel.c
	)

Set(Headers
    ${config_header_name}
	arithmeticCompiler.h
	ASTIndexNameNode.h
	charBuffer.h
	compiler.h
	cvodeData.h
	cvodeSolver.h
	daeSolver.h
	drawGraph.h
	exportdefs.h
	integratorInstance.h
	integratorSettings.h
	interpol.h
	modelSimplify.h
	nullSolver.h
	odeConstruct.h
	odeModel.h
	odeSolver.h
	processAST.h
	sbml.h
	sbmlResults.h
	sensSolver.h
	solverError.h
	util.h
	variableIndex.h
	)

# Add prefix with complete path to the header files
ADD_PREFIX(${PROJECT_SOURCE_DIR}/include/sbmlsolver/ Headers)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    Include_Directories(
            ${LIBSBML_INSTALL_DIR}/include
            ${SUNDIALS_INSTALL_DIR}/include                
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/sbmlsolver
        )
        
    Link_Directories(
            ${LIBSBML_INSTALL_DIR}/lib
            ${SUNDIALS_INSTALL_DIR}/lib
            )

    # Link_Directories(
            # ${PROJECT_SOURCE_DIR}/sbml/lib
            # ${PROJECT_SOURCE_DIR}/sundials_alt/lib
            # )
    SET(Libraries libsbml sundials_cvodes sundials_ida sundials_kinsol sundials_nvecserial)    
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    
    Include_Directories(
            ${LIBSBML_INSTALL_DIR}/include
            ${SUNDIALS_INSTALL_DIR}/include
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/sbmlsolver
            )
        
        
    Link_Directories(
            ${LIBSBML_INSTALL_DIR}/lib
            ${SUNDIALS_INSTALL_DIR}/lib
            )

    SET(Libraries sbml sundials_cvodes sundials_ida sundials_kinsol sundials_nvecserial dl)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)





# SET(Libraries  cvodes ida kinsol nvec_ser libsbml)

# Build the static library
IF(BUILD_STATIC_LIB)

  # Add the build target for the static sbmlsolver library
  ADD_LIBRARY(sbmlsolver_static STATIC ${Sources})
  TARGET_LINK_LIBRARIES(sbmlsolver_static ${Libraries})

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(sbmlsolver_static
    PROPERTIES OUTPUT_NAME sbmlsolver CLEAN_DIRECT_OUTPUT 1)

    
  # Install the sbmlsolver library
  INSTALL_TARGETS(/lib sbmlsolver_static) # this will install *.lib in the lib directory and *.dll in the bin directory
  
ENDIF(BUILD_STATIC_LIB)


# Build the shared library
IF(BUILD_SHARED_LIB)

  # Add the build target for the shared sbmlsolver library
  ADD_LIBRARY(sbmlsolver_shared SHARED ${Sources})
  TARGET_LINK_LIBRARIES(sbmlsolver_shared ${Libraries})

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(sbmlsolver_shared
    PROPERTIES OUTPUT_NAME sbmlsolver CLEAN_DIRECT_OUTPUT 1)

  # Install the sbmlsolver library
  INSTALL_TARGETS(/lib sbmlsolver_shared) # this will install *.lib in the lib directory and *.dll in the bin directory
  

ENDIF(BUILD_SHARED_LIB)


# Install the sbmlsolver header files

INSTALL(FILES ${Headers} DESTINATION "@CMAKE_INSTALL_PREFIX@/include/sbmlsolver")






