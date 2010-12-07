cmake_minimum_required (VERSION 2.6)

project (bionetsolver)

OPTION(BUILD_STATIC_LIB "Build static library" OFF)
OPTION(BUILD_SHARED_LIB "Build shared library" ON)


OPTION(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

SET(LIBSBML_INSTALL_DIR "/usr/local/libsbml-3.4.1" CACHE PATH "SBML Library Installation Directory")
SET(SUNDIALS_INSTALL_DIR "/usr/local/sundials-2.3.0" CACHE PATH "Sundials Installation Directory")
SET(CMAKE_INSTALL_PREFIX CACHE PATH "${PROJECT_SOURCE_DIR}/install")


ADD_DEFINITIONS(-DSBML_ODESOLVER_EXPORTS)

MACRO(ADD_PREFIX prefix rootlist)
  SET(outlist )
  FOREACH(root ${${rootlist}})
    LIST(APPEND outlist ${prefix}${root})
  ENDFOREACH(root)
  SET(${rootlist} ${outlist})
ENDMACRO(ADD_PREFIX)

Add_Subdirectory(source)


if(BUILD_PYTHON_BINDINGS)
	Add_Subdirectory(python)
endif(BUILD_PYTHON_BINDINGS)



if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    INSTALL(DIRECTORY  "${LIBSBML_INSTALL_DIR}/bin/" DESTINATION bin
        PATTERN "*.svn" EXCLUDE
        PATTERN "${LIBSBML_INSTALL_DIR}/bin/*.dll")

        
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/BionetSolverExamples" DESTINATION @CMAKE_INSTALL_PREFIX@/Demos
        PATTERN "*.svn" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/BionetSolverExamples/*")

    
    INSTALL(
        FILES
            ${CMAKE_SOURCE_DIR}/python/bionetAPI.py
        DESTINATION
            ${CMAKE_INSTALL_PREFIX}/PythonSetupScripts )
    
else ( ${CMAKE_SYSTEM_NAME} STREQUAL Windows)


    INSTALL(DIRECTORY  "${LIBSBML_INSTALL_DIR}/lib/" DESTINATION @CMAKE_INSTALL_PREFIX@/lib
        PATTERN "*.svn" EXCLUDE
        PATTERN "${LIBSBML_INSTALL_DIR}/lib/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/BionetSolverExamples" DESTINATION @CMAKE_INSTALL_PREFIX@/Demos
        PATTERN "*.svn" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/BionetSolverExamples/*")
        
    INSTALL(
        FILES
            ${CMAKE_SOURCE_DIR}/python/bionetAPI.py
        DESTINATION
            ${CMAKE_INSTALL_PREFIX}/pythonSetupScripts )


endif( ${CMAKE_SYSTEM_NAME} STREQUAL Windows)





