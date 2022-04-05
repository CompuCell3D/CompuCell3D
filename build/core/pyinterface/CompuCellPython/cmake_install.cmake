# Install script for directory: /N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/core/pyinterface/CompuCellPython

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp" TYPE MODULE FILES "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/pyinterface/CompuCellPython/_CompuCell.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so"
         OLD_RPATH "/N/u/mehtasau/Carbonate/miniconda3/envs/cc3denv/lib:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/steppables/BiasVectorSteppable:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ImplicitMotility:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/pyinterface/PyPlugin:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Chemotaxis:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/FocalPointPlasticity:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/PlasticityTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Plasticity:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/VolumeTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/CenterOfMass:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Mitosis:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/steppables/Mitosis:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ClusterSurface:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ClusterSurfaceTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ElasticityTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Elasticity:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ContactLocalFlex:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ContactLocalProduct:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ContactMultiCad:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/AdhesionFlex:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ConnectivityLocalFlex:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ConnectivityGlobal:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/LengthConstraint:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/BoundaryMonitor:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/MomentOfInertia:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Secretion:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/CellOrientation:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/PolarizationVector:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/Polarization23:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/OrientedGrowth:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/ContactOrientation:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/steppables/CleaverMeshDumper:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/CurvatureCalculator:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/PixelTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/NeighborTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/plugins/BoundaryPixelTracker:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/Automaton:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/Potts3D:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/PublicUtilities:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/PublicUtilities/Units:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/Boundary:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/CompuCell3D/Field3D:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/muParser/ExpressionEvaluator:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/XMLUtils:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/BasicUtils:/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/muParser:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/_CompuCell.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp" TYPE FILE FILES "/N/u/mehtasau/Carbonate/CC3D_PY3_GIT/CompuCell3D/CompuCell3D/build/core/pyinterface/CompuCellPython/CompuCell.py")
endif()

