SET(LIBS
   XMLUtilsShared
   ${PYTHON_LIBRARIES}

)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
    ${COMPUCELL3D_SOURCE_DIR}/core/XMLUtils
    # ${COMPUCELL3D_SOURCE_DIR}/BasicUtils
)

SET_SOURCE_FILES_PROPERTIES(CC3DXML.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(CC3DXML python CC3DXML.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_CC3DXML PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(CC3DXML ${LIBS})

install(TARGETS _CC3DXML    DESTINATION lib/python )
# install(TARGETS _CC3DXML    DESTINATION lib/pythonOld )

set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/CC3DXML)

INSTALL(FILES 
    ${python_files_path}/CC3DXML.py
	DESTINATION lib/python
	)

# INSTALL(FILES 
    # ${python_files_path}/CC3DXML.py
	# DESTINATION lib/pythonOld
	# )
