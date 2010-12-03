


Set(Sources
    soslib_OdeModel.cpp
    soslib_CvodeSettings.cpp
    soslib_IntegratorInstance.cpp
    BionetworkSBML.cpp
    BionetworkTemplateLibrary.cpp
    Bionetwork.cpp
    BionetworkUtilManager.cpp 	
    )

Set(Headers
    soslib_OdeModel.h
    soslib_CvodeSettings.h
    soslib_IntegratorInstance.h
    BionetworkSBML.h
    BionetworkTemplateLibrary.h
    Bionetwork.h
    BionetworkUtilManager.h
    BionetworkDLLSpecifier.h
	)

# Add prefix with complete path to the header files
ADD_PREFIX(${PROJECT_SOURCE_DIR}/include/bionetsolver/ Headers)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    Include_Directories(
            ${LIBSBML_INSTALL_DIR}/include
            ${SUNDIALS_INSTALL_DIR}/include
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/sbmlsolver
            ${PROJECT_SOURCE_DIR}/include/bionetsolver
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
            ${PROJECT_SOURCE_DIR}/include/bionetsolver
            )
        
        
    Link_Directories(
            ${LIBSBML_INSTALL_DIR}/lib
            ${SUNDIALS_INSTALL_DIR}/lib
            )

#     SET(Libraries sbml sundials_cvodes sundials_ida sundials_kinsol sundials_nvecserial dl)
       
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)





# SET(Libraries  cvodes ida kinsol nvec_ser libsbml)

# Build the static library
IF(BUILD_STATIC_LIB)

  # Add the build target for the static soslib library
  SET(Libraries soslib_static) 
  ADD_LIBRARY(bionetsolver_static STATIC ${Sources})
  TARGET_LINK_LIBRARIES(bionetsolver_static ${Libraries})

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(bionetsolver_static
    PROPERTIES OUTPUT_NAME bionetsolver CLEAN_DIRECT_OUTPUT 1)

  # Install the soslib library
  # # # INSTALL(TARGETS soslib_static DESTINATION ${PROJECT_SOURCE_DIR}/../soslib-install/bin/static)  
  # INSTALL(TARGETS soslib_static DESTINATION "@CMAKE_INSTALL_PREFIX@/lib)")
  INSTALL_TARGETS(/lib bionetsolver_static) # this will install *.lib in the lib directory and *.dll in the bin directory
  
ENDIF(BUILD_STATIC_LIB)


# Build the shared library
IF(BUILD_SHARED_LIB)

  # Add the build target for the shared soslib library
  SET(Libraries soslib_shared) 
  ADD_LIBRARY(bionetsolver_shared SHARED ${Sources})
  TARGET_LINK_LIBRARIES(bionetsolver_shared ${Libraries})

  # Set the library name and make sure it is not deleted
  SET_TARGET_PROPERTIES(bionetsolver_shared
    PROPERTIES OUTPUT_NAME bionetsolver CLEAN_DIRECT_OUTPUT 1)

  # Install the soslib library
  INSTALL_TARGETS(/lib bionetsolver_shared) # this will install *.lib in the lib directory and *.dll in the bin directory
  

ENDIF(BUILD_SHARED_LIB)


# Install the soslib header files

INSTALL(FILES ${Headers} DESTINATION "@CMAKE_INSTALL_PREFIX@/include/bionetsolver")






