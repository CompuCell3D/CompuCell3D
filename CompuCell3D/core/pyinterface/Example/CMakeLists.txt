SET(SRCS
   PyNewPlugin.cpp
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(ExampleClass ${SRCS})
    target_link_libraries(ExampleClass  ${Python3_LIBRARIES})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(ExampleClass ${SRCS})
    target_link_libraries(ExampleClass  ${Python3_LIBRARIES})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  ExampleClass)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   ExampleClass
   ${Python3_LIBRARIES}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS
   ExampleClass
   ${Python3_LIBRARIES}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/Example
   ${Python3_INCLUDE_PATH}
)

SET_SOURCE_FILES_PROPERTIES(Example.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(Example python Example.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_Example PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SWIG_LINK_LIBRARIES(Example ${LIBS})


install(TARGETS _Example    DESTINATION lib/python )
# install(TARGETS _Example    DESTINATION lib/pythonOld )


set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/Example)

INSTALL(FILES ${python_files_path}/Example.py
	      DESTINATION lib/python)

# INSTALL(FILES ${python_files_path}/Example.py
	      # DESTINATION lib/pythonOld)
          
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadStatic)
#SWIG_LINK_LIBRARIES(PlayerPython ${LIBS} simthreadShared)
