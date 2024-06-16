# Recurse into subprojects
# ADD_SUBDIRECTORY(BabySim)
ADD_SUBDIRECTORY(Field3D)
ADD_SUBDIRECTORY(Automaton)
ADD_SUBDIRECTORY(Boundary)
ADD_SUBDIRECTORY(Potts3D)
ADD_SUBDIRECTORY(plugins)
ADD_SUBDIRECTORY(steppables)

SET(QtWrappersLibs )
if(BUILD_QT_WRAPPERS)
  ADD_DEFINITIONS(-DQT_WRAPPERS_AVAILABLE)
  SET(QtWrappersLibs StreamRedirectorsShared)
endif(BUILD_QT_WRAPPERS)    


SET(SRCS
  PluginInfo.cpp
  PluginManager.cpp
  Simulator.cpp
  ClassRegistry.cpp
  DerivedProperty.cpp
  RandomNumberGenerators.cpp
  PottsParseData.cpp
  ParserStorage.cpp
  CC3DExceptions.cpp
  ExtraMembers.cpp
)

SET(INCLUDE_DIRS
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
)

set(LIBS
    cc3d::Logger
    cc3d::PublicUtilities
    cc3d::Automaton
    cc3d::Boundary
    cc3d::Field3D
    cc3d::Potts3D
)

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
endif()

ADD_SHARED_LIBRARY(CompuCellLib ${SRCS} LINK_LIBRARIES ${LIBS} cc3d::XMLUtils ${QtWrappersLibs})

TARGET_INCLUDE_DIRECTORIES(${CompuCellLibLib} PUBLIC ${INCLUDE_DIRS})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  
  if(BUILD_CPP_ONLY_EXECUTABLE)
      ADD_EXECUTABLE(CompuCell3DCML main.cpp)
      TARGET_INCLUDE_DIRECTORIES(CompuCell3DCML PUBLIC ${INCLUDE_DIRS})
      TARGET_LINK_LIBRARIES(CompuCell3DCML cc3d::CompuCellLib cc3d::XMLParserExpatLib)
      INSTALL_TARGETS(/bin CompuCell3DCML)
  else(BUILD_CPP_ONLY_EXECUTABLE)

      ADD_EXECUTABLE(mainSimpleDemo mainSimpleDemo.cpp)
      TARGET_INCLUDE_DIRECTORIES(mainSimpleDemo PUBLIC ${INCLUDE_DIRS})
      TARGET_LINK_LIBRARIES(mainSimpleDemo cc3d::CompuCellLib)
      INSTALL_TARGETS(/bin mainSimpleDemo)

  endif(BUILD_CPP_ONLY_EXECUTABLE)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  
  if(BUILD_CPP_ONLY_EXECUTABLE)
    ADD_EXECUTABLE(CompuCell3DCML main.cpp ../PublicUtilities/WindowsGlob.cpp ../PublicUtilities/WindowsGlob.h Boundary/Boundary.cpp)
    TARGET_INCLUDE_DIRECTORIES(CompuCell3DCML PUBLIC ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(CompuCell3DCML cc3d::CompuCellLib ${LIBS})
    INSTALL_TARGETS(/bin CompuCell3DCML)

  endif(BUILD_CPP_ONLY_EXECUTABLE)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D .h
  ClassRegistry
  CompuCellLibDLLSpecifier  
  DerivedProperty
  ExtraMembers
  ParseData
  Parser
  ParserStorage
  Plugin
  PluginBase
  PluginInfo
  PluginManager
  PottsParseData
  RandomNumberGenerators
  Serializer
  SimObject
  Simulator
  SteerableObject
  Steppable
  CC3D
  CC3D_Main
  CC3D_STL
  CC3DEvents
  CC3DExceptions

)
