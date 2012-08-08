# the variable "qt_app_SRCS" contains all .cpp files of this project

set(EXTRA_LINUX_LIBS) #default choice

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
   set(EXTRA_LINUX_LIBS util)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

#MESSAGE(" THIS IS EXTRALIB FOR LINUX ${EXTRA_LINUX_LIBS}")

IF(COMPUCELL3D_TEST)
  INCLUDE_DIRECTORIES(".")
  ADD_SUBDIRECTORY(Testing)
ENDIF(COMPUCELL3D_TEST)

set(SRCS
    mainCC3D.cpp
    PyScriptRunner.cpp
    simthreadAccessor.cpp
    SimthreadBase.cpp
    GraphicsDataFields.cpp
 )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_SHARED_LIBRARY(simthread ${SRCS} LINK_LIBRARIES CompuCellLibShared PlayerSettingsShared BoundaryShared Potts3DStatic AutomatonStatic BasicUtilsStatic Field3DStatic ${QT_LIBRARIES} ${PYTHON_LIBRARIES} ${EXTRA_LINUX_LIBS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    ADD_SHARED_LIBRARY(simthread ${SRCS} LINK_LIBRARIES CompuCellLibShared PlayerSettingsShared BoundaryShared Potts3DShared AutomatonShared BasicUtilsShared Field3DShared ${QT_LIBRARIES} ${PYTHON_LIBRARIES} ${EXTRA_LINUX_LIBS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH} ${XERCESC_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH} )

INSTALL_FILES(/include/CompuCellPlayer/simthread .h
  GraphicsDataFields.h  
  mainCC3D.h
  PyScriptRunner.h       
  simthreadAccessor.h
  SimthreadBase.h
  simthreadDLLSpecifier.h
)

