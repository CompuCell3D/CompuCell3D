# SET(NUMPY_INCLUDE_DIR )
# EXEC_PROGRAM(${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/PlayerPythonNew ARGS NumpyHeader.py
      # OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)






if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
SET(LIBS
  CompuCellLibShared
  #AutomatonStatic
  #BoundaryStatic
  #Field3DStatic
  #Potts3DStatic

   # XercesUtilsStatic
   BasicUtilsStatic
   PublicUtilitiesStatic
   UnitsStatic
   # XMLCerealStatic
   # ${XERCESC_LIBRARY}
   ${Python3_LIBRARIES}

)
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(/usr/include)

SET(DOLFIN_LIBS )                                                                                                
                                                                                                                 
if(USE_DOLFIN)                                                                                                   
    SET(DOLFIN_LIBS dolfin)                                                                                      
endif() 


SET(LIBS
  CompuCellLibShared
  #AutomatonShared
  #BoundaryShared
  #Field3DShared
  #Potts3DShared

   # XercesUtilsShared
   BasicUtilsShared
   PublicUtilitiesShared
   UnitsShared
   # XMLCerealShared
   # ${XERCESC_LIBRARY}
   ${Python3_LIBRARIES}
   ${DOLFIN_LIBS}

)


endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)



SET(LIBS_AUX   )

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
    ${COMPUCELL3D_SOURCE_DIR}/core/pyinterface/PyPlugin
    # ${NUMPY_INCLUDE_DIR}
    # ${COMPUCELL3D_SOURCE_DIR}/BasicUtils
    # ${COMPUCELL3D_SOURCE_DIR}/XercescUtils
    # ${COMPUCELL3D_SOURCE_DIR}/XMLCereal
    # ${XERCESC_INCLUDE_DIR}
    #${PYINTERFACE_SRC_DIR}/PyPlugin
)

# configure compilation of modules for old and new player. Old player modules are compiled with ("threads"=0) option and new are compiled with ("threads"=1) option
# set(thread_option \(\"threads\"=1\))
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CompuCell.i.in ${CMAKE_CURRENT_SOURCE_DIR}/CompuCell.i)

# set(thread_option )
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CompuCell.i.in ${CMAKE_CURRENT_SOURCE_DIR}/CompuCellOld.i)




SET_SOURCE_FILES_PROPERTIES(Fields.i  PROPERTIES CPLUSPLUS ON)
# SET_SOURCE_FILES_PROPERTIES(CompuCellPython.i CompuCellAuxPython.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(Fields python Fields.i)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_Fields PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(Fields ${LIBS} ${LIBS_AUX})



install_targets(/lib/python _Fields)

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/Fields)

#THIS IS EXAMPLE OF WORKING COMMAND 
# ADD_CUSTOM_COMMAND(
#     OUTPUT CompuCell.py.mod
#     COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pyinterface/swigmodifier.py ${python_files_path}/CompuCell.py
#     DEPENDS _CompuCell
# )
# 
# ADD_CUSTOM_TARGET(_swigmodifier ALL DEPENDS CompuCell.py.mod)

INSTALL(FILES ${python_files_path}/Fields.py 
	      DESTINATION lib/python)

# add_subdirectory(CompuCellPythonOld)

