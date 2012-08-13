

SET(SRCS
  unit_calculator.c  
  unit_calculator.lex.c  
  unit_calculator.tab.c  
  unitParserException.cpp
  Unit.cpp
  )

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

INCLUDE_DIRECTORIES (     
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/PublicUtilities/Units
    )


    
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Units ${SRCS} LINK_LIBRARIES  )
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Units ${SRCS} LINK_LIBRARIES  BasicUtilsShared)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/PublicUtilities/Units .h
  Unit
  unit_calculator
  unit_calculator_main_lib
  unit_calculator_globals
  unitParserException
  )