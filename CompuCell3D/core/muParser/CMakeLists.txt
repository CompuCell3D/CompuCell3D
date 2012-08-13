# Recurse into subprojects

SET(SRCS
  muParser.cpp
  muParserBase.cpp
  muParserBytecode.cpp
  muParserCallback.cpp
  muParserComplex.cpp
  muParserError.cpp
  muParserInt.cpp
  muParserTokenReader.cpp
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core/muParser
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  ADD_STATIC_LIBRARY(muParser ${SRCS}  )
 
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    ADD_SHARED_LIBRARY(muParser ${SRCS}  )
    
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/muParser .h
muParser
muParserBase
muParserBytecode
muParserCallback
muParserComplex
muParserDef
muParserDLL
muParserError
muParserFixes
muParserInt
muParserStack
muParserTest
muParserToken
muParserTokenReader
  )

#INCLUDE_DIRECTORIES(${XERCESC_INCLUDE_DIR})
