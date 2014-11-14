# I don't really know why these two lines are needed!
# # # CMAKE_MINIMUM_REQUIRED(VERSION 2.2)
# # # PROJECT(PYINTERFACE)

# # # include(${CMAKE_CURRENT_SOURCE_DIR}/FindSWIG.cmake) # this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX

# # # FIND_PACKAGE(SWIG REQUIRED)

# # # # INCLUDE(${SWIG_USE_FILE})
# # # INCLUDE(UseSWIG)

# # # FIND_PACKAGE(PythonLibs)
# # # FIND_PACKAGE(PythonInterp)


# # # SET(NUMPY_INCLUDE_DIR )
# # # # searching for numpy include dir
# # # EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/CAPython ARGS NumpyHeader.py
      # # # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)


INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIR})


SET(PYINTERFACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pyinterface)
SET(CMAKE_SWIG_FLAGS "")



    # SET(LIBS_AUX CenterOfMassMonitorShared)
    
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
    
    SET_SOURCE_FILES_PROPERTIES(CoreObjects.i  PROPERTIES CPLUSPLUS ON)
    # SET_SOURCE_FILES_PROPERTIES(CompuCellPython.i CompuCellAuxPython.i PROPERTIES SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(CoreObjects python CoreObjects.i)


    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
      SET_TARGET_PROPERTIES(_CoreObjects PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    SWIG_LINK_LIBRARIES(CoreObjects ${LIBS} ${LIBS_AUX})
    #TARGET_LINK_LIBRARIES(_CompuCell ${LIBS})


    install_targets(/lib/python _CoreObjects)

    set(python_files_path ${CMAKE_BINARY_DIR}/core/CAPython)
    
    INSTALL(FILES ${python_files_path}/CoreObjects.py DESTINATION lib/python) 
    
    SET_PROPERTY(TARGET _CoreObjects  PROPERTY FOLDER "CA/CoreObjects") 




