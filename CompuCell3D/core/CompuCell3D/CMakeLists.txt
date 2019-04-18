# Recurse into subprojects
# ADD_SUBDIRECTORY(BabySim)
ADD_SUBDIRECTORY(Field3D)
ADD_SUBDIRECTORY(Automaton)
ADD_SUBDIRECTORY(Boundary)
ADD_SUBDIRECTORY(Potts3D)
ADD_SUBDIRECTORY(plugins)
ADD_SUBDIRECTORY(steppables)




# FIND_PACKAGE(PythonLibs)
# INCLUDE(FindPythonLibs)




SET(QtWrappersLibs )
if(BUILD_QT_WRAPPERS)
    ADD_DEFINITIONS(-DQT_WRAPPERS_AVAILABLE)        
    SET(QtWrappersLibs StreamRedirectorsShared)
endif(BUILD_QT_WRAPPERS)    




SET(SRCS
  PluginManager.cpp
  Simulator.cpp
  ClassRegistry.cpp
  PottsParseData.cpp
  ParserStorage.cpp
  # CustomStreamBuffers.cpp
  # ${MOC_FILES}
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  ADD_SHARED_LIBRARY(CompuCellLib ${SRCS} LINK_LIBRARIES
    BasicUtilsStatic
 #   XercesUtilsStatic
#    XMLCerealStatic
    PublicUtilitiesStatic
    UnitsStatic
    AutomatonStatic
    BoundaryShared
    Field3DStatic
    Potts3DStatic
    XMLUtilsShared
    ${QtWrappersLibs}
    # ${PYTHON_LIBRARIES} #added PYTHON LIBRARIES FOR NUMPY FIELD ARRAY INTERFACE
    # ${QT_LIBS}
  )
  
    if(BUILD_CPP_ONLY_EXECUTABLE)
          ADD_EXECUTABLE(CompuCell3DCML main.cpp)
          TARGET_LINK_LIBRARIES(CompuCell3DCML
            CompuCellLibShared
            XMLParserExpatLibShared
          )
         INSTALL_TARGETS(/bin CompuCell3DCML)
     else(BUILD_CPP_ONLY_EXECUTABLE)

          ADD_EXECUTABLE(mainSimpleDemo mainSimpleDemo.cpp)
          TARGET_LINK_LIBRARIES(mainSimpleDemo
            CompuCellLibShared            
          )
         INSTALL_TARGETS(/bin mainSimpleDemo)

    endif(BUILD_CPP_ONLY_EXECUTABLE) 
    

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_SHARED_LIBRARY(CompuCellLib ${SRCS} LINK_LIBRARIES

  BasicUtilsShared
 # XercesUtilsShared
 # XMLCerealShared
  PublicUtilitiesShared
  UnitsShared
  AutomatonShared
  BoundaryShared
  Field3DShared
  Potts3DShared
  XMLUtilsShared
  ${QtWrappersLibs}
  # ${PYTHON_LIBRARIES} #added PYTHON LIBRARIES FOR NUMPY FIELD ARRAY INTERFACE
#   ${QT_LIBS}
  )

  
    if(BUILD_CPP_ONLY_EXECUTABLE)
          ADD_EXECUTABLE(CompuCell3DCML main.cpp)
          TARGET_LINK_LIBRARIES(CompuCell3DCML
            CompuCellLibShared
            BasicUtilsShared
           # XercesUtilsShared
            #XMLCerealShared
            PublicUtilitiesShared
            UnitsShared
            AutomatonShared
            BoundaryShared
            Field3DShared
            Potts3DShared
            XMLParserExpatLibShared
          )
         INSTALL_TARGETS(/bin CompuCell3DCML)
         
     else(BUILD_CPP_ONLY_EXECUTABLE)
     
        #   ADD_EXECUTABLE(CompuCell3D mainSimple.cpp)

        #   TARGET_LINK_LIBRARIES(CompuCell3D
        #     CompuCellLibShared
        #     BasicUtilsShared
        #    # XercesUtilsShared
        #     #XMLCerealShared
        #     PublicUtilitiesShared
        #     UnitsShared
        #     AutomatonShared
        #     BoundaryShared
        #     Field3DShared
        #     Potts3DShared
        #   )

        # INSTALL_TARGETS(/bin CompuCell3D)


    endif(BUILD_CPP_ONLY_EXECUTABLE) 
  
  

INSTALL_TARGETS("/${COMPUCELL3D_INSTALL_LIB_DIR}"
  # /lib
  CompuCellLibShared)

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/CompuCell3D .h
  ClassRegistry
  CompuCellLibDLLSpecifier  
  ParseData
  Parser
  ParserStorage
  Plugin
  PluginBase
  PluginManager
  PottsParseData
  Serializer
  SimObject
  Simulator
  SteerableObject
  Steppable
  CC3D
  CC3D_Main
  CC3D_STL
  CC3DEvents
  
  )

#INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})
