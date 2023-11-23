
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(PYINTERFACE)

# this is slightly modified FindSWIG.cmake file -  had to tune it to work with OSX
#include(${CMAKE_CURRENT_SOURCE_DIR}/FindSWIG.cmake)

FIND_PACKAGE(SWIG REQUIRED)

INCLUDE(UseSWIG)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
INCLUDE_DIRECTORIES(${Python3_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(PYINTERFACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")

IF(SWIG_FOUND)
    MESSAGE("SWIG_DIR=" ${SWIG_DIR})
    add_subdirectory(CompuCellExtraModules)

ELSE(SWIG_FOUND)
    message("could not find SWIG in your system")
ENDIF(SWIG_FOUND)

