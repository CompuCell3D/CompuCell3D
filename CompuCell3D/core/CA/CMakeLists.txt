
SET(SRCS
  CAManager.cpp
  CACellStack.cpp
  CACellInventory.cpp
  )

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(PythonInterp)
  
# here we setup SWIG for all subdirectories  
include(${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/FindSWIG.cmake) # this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX

FIND_PACKAGE(SWIG REQUIRED)
# INCLUDE(${SWIG_USE_FILE})
INCLUDE(UseSWIG)  # without this inclusion you get errorlike unresolved external symbol __DllMainCRTStartup@12 on windows - probably other cryptic errors on other platforms
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(PythonInterp)  


  
INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )    

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(CA ${SRCS} LINK_LIBRARIES BoundaryShared Field3DStatic)
   # ADD_SHARED_LIBRARY(CA ${SRCS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   # ADD_SHARED_LIBRARY(CA ${SRCS})
   ADD_SHARED_LIBRARY(CA ${SRCS} LINK_LIBRARIES BoundaryShared BasicUtilsShared Field3DShared)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/CA/CAManager .h
  CAManager
  CACell
  CACellStack
  )

ADD_SUBDIRECTORY(modules) # those are equivalents of old style plugins

  
# # # ADD_EXECUTABLE(CACL main.cpp)
# # # TARGET_LINK_LIBRARIES(CACL CAShared ${PYTHON_LIBRARIES})
# # # INSTALL_TARGETS(/bin CACL)  
# # # SET_PROPERTY(TARGET CACL  PROPERTY FOLDER "CA/CA")


#folder setup for better display iof projects in the IDE's          
SET_PROPERTY(TARGET CAShared  PROPERTY FOLDER "CA/CA")




IF(SWIG_FOUND)

    SET(NUMPY_INCLUDE_DIR )
    # searching for numpy include dir
    EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/CA/CAPython ARGS NumpyHeader.py
          OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)  

    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
    INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
    INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIR})
    
    INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMPUCELL3D_SOURCE_DIR}/core
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
        # # # ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin
 
    )  

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
    
    SET_PROPERTY(TARGET _CA  PROPERTY FOLDER "CA/CA")    
    
    add_subdirectory(CAPython)
    
ELSE(SWIG_FOUND)
    message("could not find SWIG in your system")
ENDIF(SWIG_FOUND)