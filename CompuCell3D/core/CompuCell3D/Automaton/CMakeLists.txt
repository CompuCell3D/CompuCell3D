SET(SRCS
  Automaton.cpp
  CellType.cpp
  Transition.cpp)

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_STATIC_LIBRARY(Automaton ${SRCS})
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   ADD_SHARED_LIBRARY(Automaton ${SRCS})
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

INSTALL_FILES(/include/CompuCell3D/CompuCell3D/Automaton .h
  Automaton
  CellTypeDynamicClassNode
  CellType
  CellTypeNode
  Transition
  CC3D_Automaton
  )
