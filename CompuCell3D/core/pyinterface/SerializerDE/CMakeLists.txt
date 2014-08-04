# Recurse into subprojects

SET(SRCS
  SerializerDE.cpp
  )



  
# FIND_PACKAGE(VTK REQUIRED)
# message("VTK_MAJOR_VERSION=" ${VTK_MAJOR_VERSION})

if (${VTK_MAJOR_VERSION} EQUAL "6")    
    SET(VTK_LIBS vtkRenderingOpenGL vtkInteractionStyle vtkRenderingVolumeOpenGL vtkIOLegacy)
endif()

if (${VTK_MAJOR_VERSION} EQUAL "5")    
    SET(VTK_LIBS vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)
endif()


INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${VTK_INCLUDE_DIRS}
   ${COMPUCELL3D_SOURCE_DIR}/core
   ${COMPUCELL3D_SOURCE_DIR}/core/Utils
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
   ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins

    )

LINK_DIRECTORIES(${VTK_LIBRARY_DIRS})
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS
       CompuCellLibShared
       BasicUtilsStatic
       FieldExtractor
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS}
       # simthreadShared

    )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    
    
    SET(LIBS
       CompuCellLibShared
       BasicUtilsShared       
       FieldExtractor
       ${PYTHON_LIBRARIES}
       ${VTK_LIBS} 
    )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  
INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core/SerializerDE
    )

    
    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE SHARED ${SRCS})
    
    target_link_libraries(SerializerDE  ${LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_LIBRARY(SerializerDE ${SRCS})

    target_link_libraries(SerializerDE  ${LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_TARGETS(/lib
  SerializerDE)
    
    
INSTALL_FILES(/include/SerializerDE .h
SerializerDE
)


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   SerializerDE
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(LIBS_SWIG
   SerializerDE
   ${LIBS}
   ${PYTHON_LIBRARIES}

)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PYINTERFACE_SRC_DIR}/SerializerDE
   ${PYTHON_INCLUDE_PATH}
   ${VTK_INCLUDE_DIRS}
)

SET_SOURCE_FILES_PROPERTIES(SerializerDEPy.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(SerializerDEPy python SerializerDEPy.i)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  SET_TARGET_PROPERTIES(_SerializerDEPy PROPERTIES SUFFIX ".pyd") # changes dll name to pyd sop that it is compatible with new Python standard
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# SET_TARGET_PROPERTIES(_PlayerPythonNew PROPERTIES OUTPUT_NAME  "_PlayerPython")   
    
SWIG_LINK_LIBRARIES(SerializerDEPy ${LIBS_SWIG})

install_targets(/lib/python _SerializerDEPy)
set(python_files_path ${CMAKE_BINARY_DIR}/core/pyinterface/SerializerDE)

INSTALL(FILES ${python_files_path}/SerializerDEPy.py
	      DESTINATION lib/python)


#INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})
