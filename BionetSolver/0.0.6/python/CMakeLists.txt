


CMAKE_MINIMUM_REQUIRED(VERSION 2.2)

# this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX
## include(${CMAKE_CURRENT_SOURCE_DIR}/FindSWIG.cmake) 

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(UseSWIG)

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(PythonInterp)




if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    Include_Directories(
            ${LIBSBML_INSTALL_DIR}/include
            ${SUNDIALS_INSTALL_DIR}/include
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/sbmlsolver
            ${PYTHON_INCLUDE_PATH}
            ${CMAKE_CURRENT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/include            
            )
        
        
    Link_Directories(
            ${LIBSBML_INSTALL_DIR}/lib
            ${SUNDIALS_INSTALL_DIR}/lib
            )

 
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    
    Include_Directories(
            ${LIBSBML_INSTALL_DIR}/include
            ${SUNDIALS_INSTALL_DIR}/include
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/sbmlsolver
            ${PYTHON_INCLUDE_PATH}
            ${CMAKE_CURRENT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/include            
            )
        
        
    Link_Directories(
            ${LIBSBML_INSTALL_DIR}/lib
            ${SUNDIALS_INSTALL_DIR}/lib
            )

	
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)



SET(PYINTERFACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_SWIG_FLAGS "")

SET_SOURCE_FILES_PROPERTIES(BionetSolverPy.i  PROPERTIES CPLUSPLUS ON)
SWIG_ADD_MODULE(BionetSolverPy python BionetSolverPy.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  # changes dll name to pyd so that it is compatible with new Python standard
  SET_TARGET_PROPERTIES(_BionetSolverPy PROPERTIES SUFFIX ".pyd") 
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

IF(BUILD_STATIC_LIB)
	SWIG_LINK_LIBRARIES(BionetSolverPy bionetsolver_static ${PYTHON_LIBRARIES})
ENDIF(BUILD_STATIC_LIB)


IF(BUILD_SHARED_LIB)
	SWIG_LINK_LIBRARIES(BionetSolverPy bionetsolver_shared  sundials_cvodes sundials_nvecserial sbml dl ${PYTHON_LIBRARIES})
ENDIF(BUILD_SHARED_LIB)

#THIS IS EXAMPLE OF WORKING COMMAND
# ADD_CUSTOM_COMMAND(
#     OUTPUT BionetSolverPy.py.mod
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/swigmodifier.py ${CMAKE_BINARY_DIR}/python/BionetSolverPy.py  
#     DEPENDS _BionetSolverPy
# )
# 
# ADD_CUSTOM_TARGET(_swigmodifier ALL DEPENDS BionetSolverPy.py.mod)



INSTALL(
    TARGETS
        _BionetSolverPy
    DESTINATION
        ${CMAKE_INSTALL_PREFIX}/lib/python )
        
INSTALL(
    FILES
        ${CMAKE_BINARY_DIR}/python/BionetSolverPy.py
    DESTINATION
        ${CMAKE_INSTALL_PREFIX}/lib/python )



