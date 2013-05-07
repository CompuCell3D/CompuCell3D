# SET(NUMPY_INCLUDE_DIR )
# EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/PlayerPythonNew ARGS NumpyHeader.py
      # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
SET(LIBS
  # # # CompuCellLibShared
  # # # #AutomatonStatic
  # # # #BoundaryStatic
  # # # #Field3DStatic
  # # # #Potts3DStatic

   # # # # XercesUtilsStatic
   # # # BasicUtilsStatic
   # # # PublicUtilitiesStatic
   # # # UnitsStatic
   # # # # XMLCerealStatic
   # # # # ${XERCESC_LIBRARY}
   # rr-libstruct-static
   # roadrunner-static
   
   rr-libstruct
   roadrunner
   
   
   ${PYTHON_LIBRARIES}

)
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(/usr/include)


SET(LIBS

  
  
 # sundials_cvode
 # sundials_nvecserial
 # blas
 # lapack
 # f2c
 # pugi-static
 # nleq-static
 # rr-libstruct-static
 # poco_foundation-static  
 
  # lapack
  
  
  
  
  # blas
  # lapack  
  rr-libstruct  
  roadrunner
  # f2c
  # m
  
  
  
  
 
  # ${PYTHON_LIBRARIES} 
  # # # CompuCellLibShared
  # # # #AutomatonShared
  # # # #BoundaryShared
  # # # #Field3DShared
  # # # #Potts3DShared

   # # # # XercesUtilsShared
   # # # BasicUtilsShared
   # # # PublicUtilitiesShared
   # # # UnitsShared
   # # # # XMLCerealShared
   # # # # ${XERCESC_LIBRARY}
   # # # ${PYTHON_LIBRARIES}
   # # # dolfin

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


SET(SRCS
   RoadRunnerAdaptor.cpp
)


INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${RR_SOURCE_ROOT}
    ${THIRD_PARTY_INSTALL_FOLDER}/include
    ${THIRD_PARTY_INSTALL_FOLDER}/include/rr-libstruct
    ${THIRD_PARTY_INSTALL_FOLDER}/include/sbml
    ${THIRD_PARTY_INSTALL_FOLDER}/include/cvode   
   # ${RR_ROOT}
   # ${THIRD_PARTY_FOLDER}/sundials/include   
)

# MESSAGE (${THIRD_PARTY_FOLDER} )
# MESSAGE (${RR_ROOT})


SET(LIBS_AUX   )



if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(RoadRunnerAdaptor SHARED ${SRCS})
    
    target_link_libraries(RoadRunnerAdaptor  ${LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(RoadRunnerAdaptor ${SRCS})

    target_link_libraries(RoadRunnerAdaptor  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

set_property(TARGET RoadRunnerAdaptor  
 PROPERTY  COMPILE_DEFINITIONS 
 STATIC_LIBSTRUCT
 STATIC_RR
 PUGI_STATIC
) 



# configure compilation of modules for old and new player. Old player modules are compiled with ("threads"=0) option and new are compiled with ("threads"=1) option
# set(thread_option \(\"threads\"=1\))





SET_SOURCE_FILES_PROPERTIES(RoadRunner.i  PROPERTIES CPLUSPLUS ON)
# SET_SOURCE_FILES_PROPERTIES(CompuCellPython.i CompuCellAuxPython.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(RoadRunner python RoadRunner.i)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_RoadRunner PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard  
  
    set_property(TARGET _RoadRunner 
     PROPERTY  COMPILE_DEFINITIONS 
     SHARED_LIBSTRUCT
     SHARED_RR
     PUGI_STATIC
    )     
  
    # set_property(TARGET _RoadRunner 
     # PROPERTY  COMPILE_DEFINITIONS 
     # STATIC_LIBSTRUCT
     # STATIC_RR
     # PUGI_STATIC
    # )   
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  # # SET_TARGET_PROPERTIES(_RoadRunner PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard  
    set_property(TARGET _RoadRunner 
     PROPERTY  COMPILE_DEFINITIONS 
     SHARED_LIBSTRUCT
     # STATIC_RR
     PUGI_STATIC
    )   

    
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(RoadRunner ${LIBS} ${LIBS_AUX})



install_targets(/lib/python _RoadRunner)

set(python_files_path ${CMAKE_BINARY_DIR}/pyinterface/RoadRunner)

#THIS IS EXAMPLE OF WORKING COMMAND 
# ADD_CUSTOM_COMMAND(
#     OUTPUT CompuCell.py.mod
#     COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/swigmodifier.py ${python_files_path}/CompuCell.py   
#     DEPENDS _CompuCell
# )
# 
# ADD_CUSTOM_TARGET(_swigmodifier ALL DEPENDS CompuCell.py.mod)

INSTALL(FILES ${python_files_path}/RoadRunner.py 
	      DESTINATION lib/python)

          
message ("THIS IS BUILD CC3D" ${BUILD_CC3D_EXTENSION})          


if (${BUILD_CC3D_EXTENSION} STREQUAL ON)
    
    message("INSIDE BUILD CC3D OF PYTHON INTERFACE")
          
    INSTALL(FILES ${python_files_path}/RoadRunner.py DESTINATION ${CC3D_INSTALL_PREFIX}/lib/python)

    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        install(TARGETS _RoadRunner RUNTIME DESTINATION ${CC3D_INSTALL_PREFIX}/lib/python LIBRARY DESTINATION ${CC3D_INSTALL_PREFIX}/lib/python)          
    else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)    
        install(TARGETS _RoadRunner  LIBRARY DESTINATION ${CC3D_INSTALL_PREFIX}/lib/python)          
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)    
              
endif (${BUILD_CC3D_EXTENSION} STREQUAL ON)             

          
          
# add_subdirectory(CompuCellPythonOld)

