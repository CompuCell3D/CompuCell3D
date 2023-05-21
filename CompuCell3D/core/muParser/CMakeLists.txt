SET(SRCS
  muParser.cpp
  muParserBase.cpp
  muParserBytecode.cpp
  muParserCallback.cpp
  muParserError.cpp
  muParserInt.cpp
  muParserTokenReader.cpp
)

SET(INCLUDE_DIRS 
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/muParser>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/muParser>
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  ADD_STATIC_LIBRARY(muParser ${SRCS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  ADD_SHARED_LIBRARY(muParser ${SRCS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

TARGET_INCLUDE_DIRECTORIES(${muParserLib} PUBLIC ${INCLUDE_DIRS})

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/muParser .h
  muParser
  muParserBase
  muParserBytecode
  muParserCallback
  muParserDef
  muParserDLL
  muParserError
  muParserFixes
  muParserInt
  muParserTest
  muParserTemplateMagic
  muParserToken
  muParserTokenReader
  CC3D_muParser

)

# add_subdirectory(Test)
add_subdirectory(ExpressionEvaluator)