SET(SRCS
  CAManager.cpp
  CACellStack.cpp
  CACellInventory.cpp
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )

    
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

  
ADD_EXECUTABLE(CACL main.cpp)
TARGET_LINK_LIBRARIES(CACL CAShared)
INSTALL_TARGETS(/bin CACL)  