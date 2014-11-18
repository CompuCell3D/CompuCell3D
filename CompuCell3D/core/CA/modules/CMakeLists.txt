include(${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/FindSWIG.cmake) # this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX

FIND_PACKAGE(SWIG REQUIRED)
# INCLUDE(${SWIG_USE_FILE})
INCLUDE(UseSWIG)  # without this inclusion you get errorlike unresolved external symbol __DllMainCRTStartup@12 on windows - probably other cryptic errors on other platforms
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(PythonInterp)  

# # # SET(NUMPY_INCLUDE_DIR )
# # # # searching for numpy include dir
# # # EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/CAPython ARGS NumpyHeader.py
      # # # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)


INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

SET(MODULE_LIBS CAShared)


MACRO(ADD_CA_MODULE)
    PARSE_ARGUMENTS(MODULE
        "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS;SWIG"
        ""
        ${ARGN}
    )
    CAR(MODULE_NAME ${MODULE_DEFAULT_ARGS})
    CDR(MODULE_SOURCES ${MODULE_DEFAULT_ARGS})

    MESSAGE("*** NAME OF MODULE ${MODULE_NAME}")
    MESSAGE("Sources: ${MODULE_SOURCES}")
    MESSAGE("Link libraries: ${MODULE_LINK_LIBRARIES}")
    MESSAGE("EXTRA_COMPILER_FLAGS: ${MODULE_EXTRA_COMPILER_FLAGS}")
    MESSAGE("SWIG: ${MODULE_SWIG}")
    
    SET(USE_SWIG "ON")
    if ("${MODULE_SWIG}" STREQUAL "OFF")
        SET(USE_SWIG "OFF")
    endif ("${MODULE_SWIG}" STREQUAL "OFF")
    
    
    # SET(MODULE_NAME ${MODULE_NAME})
    file(GLOB source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx" "${CMAKE_CURRENT_SOURCE_DIR}/*.c" )
    SET(SRCS ${source_files})

    INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/Utils
    )  
  
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        SET(LIBS
           CAShared
           BasicUtilsStatic
           ${PYTHON_LIBRARIES}
        )
    else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        SET(LIBS
           CAShared
           BasicUtilsShared
           ${PYTHON_LIBRARIES}
        )
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)  
    
    
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        ADD_SHARED_LIBRARY(${MODULE_NAME} ${SRCS} LINK_LIBRARIES ${LIBS})
    else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        ADD_SHARED_LIBRARY(${MODULE_NAME} ${SRCS} LINK_LIBRARIES ${LIBS})
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    INSTALL_TARGETS(/lib ${MODULE_NAME}Shared)    
    SET_PROPERTY(TARGET ${MODULE_NAME}Shared  PROPERTY FOLDER "CA/${MODULE_NAME}/${MODULE_NAME}")
    
    if (${USE_SWIG} STREQUAL "ON")
    
        if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

        SET(LIBS_SWIG
           ${MODULE_NAME}Shared
           ${LIBS}
           ${PYTHON_LIBRARIES}

        )

        else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

        SET(LIBS_SWIG
           ${MODULE_NAME}Shared
           ${LIBS}
           ${PYTHON_LIBRARIES}

        )    
        endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        
        INCLUDE_DIRECTORIES(
           ${CMAKE_CURRENT_SOURCE_DIR}
           # # # ${PYINTERFACE_SRC_DIR}/PlayerPythonNew
           ${PYTHON_INCLUDE_PATH}
        )

        SET_SOURCE_FILES_PROPERTIES(${MODULE_NAME}.i PROPERTIES CPLUSPLUS ON)

        SWIG_ADD_MODULE(${MODULE_NAME} python ${MODULE_NAME}.i)

        if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
          SET_TARGET_PROPERTIES(_${MODULE_NAME} PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
        endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

        SET_TARGET_PROPERTIES(_${MODULE_NAME} PROPERTIES OUTPUT_NAME  "_${MODULE_NAME}")   
            
        SWIG_LINK_LIBRARIES(${MODULE_NAME} ${LIBS_SWIG})

        install_targets(/lib/python/CA _${MODULE_NAME})
        set(python_files_path ${CMAKE_BINARY_DIR}/core/CA/modules/${MODULE_NAME})

        INSTALL(FILES ${python_files_path}/${MODULE_NAME}.py
                  DESTINATION lib/python/CA)

        #folder setup for better display iof projects in the IDE's          
        
        SET_PROPERTY(TARGET _${MODULE_NAME}  PROPERTY FOLDER "CA/${MODULE_NAME}/${MODULE_NAME}")
        
    
    endif(${USE_SWIG} STREQUAL "ON")
  
  INCLUDE_DIRECTORIES ( 
        ${COMPUCELL3D_SOURCE_DIR}/core
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
  )  
  
  
  
ENDMACRO(ADD_CA_MODULE)




# add_subdirectory(CenterOfMassMonitor)  
add_subdirectory(CanonicalProbability)  
add_subdirectory(ChemotaxisProbability)
add_subdirectory(CAPDESolvers)  
add_subdirectory(CellTrail)
