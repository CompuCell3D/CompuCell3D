# Install script for directory: /N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/pyinterface/PyPlugin" TYPE FILE FILES
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/PyPlugin.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/PyCompuCellObjAdapter.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/EnergyFunctionPyWrapper.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/ChangeWatcherPyWrapper.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/StepperPyWrapper.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/TypeChangeWatcherPyWrapper.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/NeighborFinderParams.h"
    "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/PyAttributeAdder.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib" TYPE SHARED_LIBRARY FILES "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/PyPlugin/libPyPlugin.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so"
         OLD_RPATH "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/CompuCell3D:/N/u/mehtasau/Carbonate/miniconda3/envs/cc3denv/lib:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/CompuCell3D/Automaton:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/CompuCell3D/Potts3D:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/PublicUtilities:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/PublicUtilities/Units:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/CompuCell3D/Boundary:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/CompuCell3D/Field3D:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/muParser/ExpressionEvaluator:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/muParser:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/XMLUtils:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/BasicUtils:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/lib/libPyPlugin.so")
    endif()
  endif()
endif()
