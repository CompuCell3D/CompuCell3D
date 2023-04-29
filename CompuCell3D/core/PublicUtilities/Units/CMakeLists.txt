SET(SRCS
#  unit_calculator.c
#  unit_calculator.lex.c
#  unit_calculator.tab.c
#  unitParserException.cpp
  Unit.cpp
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Units ${SRCS} LINK_LIBRARIES )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Units ${SRCS} LINK_LIBRARIES )
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

TARGET_INCLUDE_DIRECTORIES(${UnitsLib} PUBLIC
   $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
   $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core/PublicUtilities/Units>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/PublicUtilities/Units>
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/PublicUtilities/Units .h
  Unit
  unit_calculator
  unit_calculator_main_lib
  unit_calculator_globals
  unitParserException
)