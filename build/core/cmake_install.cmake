# Install script for directory: /N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/N/u/mehtasau/Carbonate/cc3d_build_wdocs")
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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/BasicUtils/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/Utils/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/XMLUtils/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/PublicUtilities/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/muParser/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/Cleaver/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/pyinterface/cmake_install.cmake")
  include("/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/post_install_tasks/cmake_install.cmake")

endif()

