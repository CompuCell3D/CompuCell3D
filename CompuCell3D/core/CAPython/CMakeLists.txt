# I don't really know why these two lines are needed!
CMAKE_MINIMUM_REQUIRED(VERSION 2.2)
PROJECT(PYINTERFACE)

include(${CMAKE_CURRENT_SOURCE_DIR}/FindSWIG.cmake) # this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX

FIND_PACKAGE(SWIG REQUIRED)

# INCLUDE(${SWIG_USE_FILE})
INCLUDE(UseSWIG)

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(PythonInterp)


SET(NUMPY_INCLUDE_DIR )
# searching for numpy include dir
EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/CAPython ARGS NumpyHeader.py
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)


INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIR})


SET(PYINTERFACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pyinterface)
SET(CMAKE_SWIG_FLAGS "")

IF(SWIG_FOUND)

    SET(LIBS_AUX CenterOfMassMonitorShared)
    
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        SET(LIBS
          CAShared
          #AutomatonStatic
          #BoundaryStatic
          Field3DStatic
          BasicUtilsStatic
          PublicUtilitiesStatic
          UnitsStatic
          ${PYTHON_LIBRARIES}

        )      
    else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)    
        SET(LIBS
          CompuCellLibShared
          Field3DShared
           BasicUtilsShared
           PublicUtilitiesShared
           UnitsShared
           ${PYTHON_LIBRARIES}
           # ${DOLFIN_LIBS}
        )    
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)        
    
    
    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMPUCELL3D_SOURCE_DIR}/core
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
        # # # ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin
 
    )    
 
    SET_SOURCE_FILES_PROPERTIES(CA.i  PROPERTIES CPLUSPLUS ON)
    # SET_SOURCE_FILES_PROPERTIES(CompuCellPython.i CompuCellAuxPython.i PROPERTIES SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(CA python CA.i)


    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
      SET_TARGET_PROPERTIES(_CA PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    SWIG_LINK_LIBRARIES(CA ${LIBS} ${LIBS_AUX})
    #TARGET_LINK_LIBRARIES(_CompuCell ${LIBS})


    install_targets(/lib/python _CA)

    set(python_files_path ${CMAKE_BINARY_DIR}/core/CAPython)
    
    INSTALL(FILES ${python_files_path}/CA.py DESTINATION lib/python)    



 
ELSE(SWIG_FOUND)
    message("could not find SWIG in your system")
ENDIF(SWIG_FOUND)


# IF(SWIG_FOUND)
    # add_subdirectory(CC3DXML)
    # add_subdirectory(CompuCellPython)
    
# # if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)        
    
# # else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)        
    # # add_subdirectory(FiPyInterface)
# # endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)            

# #    add_subdirectory(PlayerPythonOld)

    # FIND_PACKAGE(VTK REQUIRED)
    # message("VTK_MAJOR_VERSION=" ${VTK_MAJOR_VERSION})
    # if (${VTK_MAJOR_VERSION} EQUAL "6")    
        # add_definitions(-DVTK6)
    # endif()    
    

    # add_subdirectory(PlayerPythonNew)
    # add_subdirectory(SerializerDE)
    # add_subdirectory(PyPlugin)
    # add_subdirectory(Example)
    # add_subdirectory(Fields)
# if(USE_DOLFIN)    
    # add_subdirectory(dolfinCC3D)
# endif(USE_DOLFIN)        
    
# ELSE(SWIG_FOUND)
    # message("could not find SWIG in your system")
# ENDIF(SWIG_FOUND)

#CMAKE_MINIMUM_REQUIRED(VERSION 2.2)
#PROJECT(PYINTERFACE)
