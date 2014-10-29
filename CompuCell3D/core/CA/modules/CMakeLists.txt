SET (SRCS CenterOfMassMonitor.cpp)

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(CenterOfMassMonitor ${SRCS} LINK_LIBRARIES CAShared)
   # ADD_SHARED_LIBRARY(CA ${SRCS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   # ADD_SHARED_LIBRARY(CA ${SRCS})
   ADD_SHARED_LIBRARY(CenterOfMassMonitor ${SRCS} LINK_LIBRARIES CAShared)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/CA/modules .h
  CenterOfMassMonitor)

# here we setup SWIG for all subdirectories  
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
# # # INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
# # # INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
# # # INCLUDE_DIRECTORIES(${NUMPY_INCLUDE_DIR})

add_subdirectory(CanonicalProbability)  
add_subdirectory(PDESolvers)  