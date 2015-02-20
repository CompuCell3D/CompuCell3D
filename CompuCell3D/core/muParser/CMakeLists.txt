# Recurse into subprojects

SET(SRCS
  muParser.cpp
  muParserBase.cpp
  muParserBytecode.cpp
  muParserCallback.cpp
  muParserError.cpp
  muParserInt.cpp
  muParserTokenReader.cpp
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core/muParser
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  ADD_STATIC_LIBRARY(muParser ${SRCS}  )
 
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    ADD_SHARED_LIBRARY(muParser ${SRCS}  )
    
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# INSTALL_FILES(/include/muParser .h
INSTALL_FILES(/include/CompuCell3D/muParser .h
muParser
muParserBase
muParserBytecode
muParserCallback
muParserDef
muParserDLL
muParserError
muParserFixes
muParserInt
muParserStack
muParserTest
muParserTemplateMagic
muParserToken
muParserTokenReader
CC3D_muParser

  )




# add_subdirectory(Test)
add_subdirectory(ExpressionEvaluator)