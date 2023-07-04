SET(SRCS
  AttributeAdder.cpp
  Cell.cpp
  CellInventory.cpp  
  DefaultAcceptanceFunction.cpp
  CustomAcceptanceFunction.cpp
  EnergyFunctionCalculator.cpp
  EnergyFunctionCalculatorStatistics.cpp
  EnergyFunctionCalculatorTestDataGeneration.cpp
  TypeTransition.cpp
  StandardFluctuationAmplitudeFunctions.cpp
  Potts3D.cpp
  PottsTestData.cpp
)

SET(GOMP_LIB )

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL Windows OR MINGW)

  # during compilation wth MINGW gomp - OPenMP dynamic library has to be linked explicitely
  SET(GOMP_LIB gomp)

endif(NOT ${CMAKE_SYSTEM_NAME} STREQUAL Windows OR MINGW)

SET(LIBS
  ${GOMP_LIB}
  cc3d::Boundary
  cc3d::Field3D
  cc3d::PublicUtilities
  cc3d::ExpressionEvaluator
)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_STATIC_LIBRARY(Potts3D ${SRCS}
    LINK_LIBRARIES
    ${LIBS}
    EXTRA_COMPILER_FLAGS
    ${OpenMP_CXX_FLAGS}
  )

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

  ADD_SHARED_LIBRARY(Potts3D ${SRCS} 
    LINK_LIBRARIES
    ${LIBS}
    EXTRA_COMPILER_FLAGS
    ${OpenMP_CXX_FLAGS}
  )

endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

TARGET_INCLUDE_DIRECTORIES(${Potts3DLib} PUBLIC
  $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/CompuCell3D/Potts3D .h
  AcceptanceFunction
  AttributeAdder
  CellChangeWatcher
  CellGChangeWatcher
  Cell
  CellInventoryWatcher
  CellInventory
  CellTypeMotilityData
  DefaultAcceptanceFunction
  CustomAcceptanceFunction
  EnergyFunction
  EnergyFunctionCalculator
  EnergyFunctionCalculatorStatistics
  EnergyFunctionCalculatorTestDataGeneration
  FirstOrderExpansionAcceptanceFunction
  FixedStepper
  Potts3D
  PottsTestData
  Stepper  
  TypeChangeWatcher
  TypeTransition
  FluctuationAmplitudeFunction
  StandardFluctuationAmplitudeFunctions
  CC3D_Potts3D
)
