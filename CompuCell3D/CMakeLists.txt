CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(COMPUCELL3D)


if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

if (${CMAKE_VERSION} VERSION_GREATER 3.1.0)
    # cmake_policy(SET CMP0053 OLD)
    cmake_policy(SET CMP0053 NEW)
endif()

##APPLE CHECK
IF (APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 " CACHE STRING "flags used during compilation by C++ compiler" FORCE)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 " CACHE STRING "flags used during compilation by C compiler" FORCE)
    SET(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "OSX deplyment target" FORCE)

    MESSAGE ("GOT APPLE")
ENDIF (APPLE)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE (OpenMP)

# Should we build shared libraries
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)


# Specify a version for all libs to use
# letters A, B, C are added to ensure nice display of version in cmake gui
SET (COMPUCELL3D_A_MAJOR_VERSION "4" CACHE STRING "Major Version")
SET (COMPUCELL3D_B_MINOR_VERSION "0" CACHE STRING "Minor Version")
SET (COMPUCELL3D_C_BUILD_VERSION "0" CACHE STRING "Build Version")


# and here we assign them back to what they should be
SET (COMPUCELL3D_MAJOR_VERSION ${COMPUCELL3D_A_MAJOR_VERSION})
SET (COMPUCELL3D_MINOR_VERSION ${COMPUCELL3D_B_MINOR_VERSION})
SET (COMPUCELL3D_BUILD_VERSION ${COMPUCELL3D_C_BUILD_VERSION})


message(" THIS IS COMPUCELL3D_BUILD_VERSION " ${COMPUCELL3D_BUILD_VERSION})
message("COMPUCELL3D_C_BUILD_VERSION is " ${COMPUCELL3D_C_BUILD_VERSION})

SET(COMPUCELL3D_VERSION
  "${COMPUCELL3D_MAJOR_VERSION}.${COMPUCELL3D_MINOR_VERSION}.${COMPUCELL3D_BUILD_VERSION}")

SET(USE_DOLFIN_OPTION OFF)
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(USE_DOLFIN_OPTION OFF)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

OPTION(USE_DOLFIN "Build modules that depend on dolfin FEM solver" ${USE_DOLFIN_OPTION})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

INCLUDE_DIRECTORIES (${COMPUCELL3D_SOURCE_DIR}/core)


OPTION(USE_LIBRARY_VERSIONS "Use version numbers in libraries" OFF)

SET(USE_LIBRARY_VERSIONS OFF)

IF(USE_LIBRARY_VERSIONS)
    SET(COMPUCELL3D_LIBRARY_PROPERTIES ${COMPUCELL3D_LIBRARY_PROPERTIES}
    VERSION "${COMPUCELL3D_VERSION}"
    SOVERSION "${COMPUCELL3D_MAJOR_VERSION}.${COMPUCELL3D_MINOR_VERSION}"
    )

ELSE(USE_LIBRARY_VERSIONS)
    SET(COMPUCELL3D_LIBRARY_PROPERTIES ${COMPUCELL3D_LIBRARY_PROPERTIES} VERSION "" SOVERSION "")
ENDIF(USE_LIBRARY_VERSIONS)

# Enable Unit Testing
OPTION(COMPUCELL3D_TEST "Unit testing of CompuCell3D" OFF)
MARK_AS_ADVANCED(COMPUCELL3D_TEST)

IF(COMPUCELL3D_TEST)
	ENABLE_TESTING()
	INCLUDE(FindCppUnit.cmake)
	INCLUDE_DIRECTORIES(${CPPUNIT_INCLUDE_DIR})
ENDIF(COMPUCELL3D_TEST)

# Always include the top level source for compiles
SET_DIRECTORY_PROPERTIES(PROPERTIES
  INCLUDE_DIRECTORIES ${COMPUCELL3D_SOURCE_DIR})

# Where to install files relative to prefix
SET(SITE_PACKAGES_INSTALL ${CMAKE_INSTALL_PREFIX}/lib/site-packages)

SET(COMPUCELL3D_INSTALL_LIB_DIR "lib/site-packages/cc3d/cpp/lib")
SET(COMPUCELL3D_INSTALL_PLUGIN_DIR " lib/site-packages/cc3d/cpp/CompuCell3DPlugins")
SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR " lib/site-packages/cc3d/cpp/CompuCell3DSteppables")
SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR_FULL_PATH ${CMAKE_INSTALL_PREFIX}/lib/site-packages/cc3d/cpp/CompuCell3DSteppables)
SET(COMPUCELL3D_INSTALL_INCLUDE_DIR "include/CompuCell3D")

SET(COMPUCELL3D_INSTALL_SWIG_MODULES_DIR "lib/site-packages/cc3d/cpp/")

# Find libz for use in the CompuCell/IO utils
INCLUDE(FindZLIB)

# Should we build wrappings
OPTION(BUILD_PYINTERFACE "Build Python language wrappings" ON)

OPTION(INSTALL_PLAYER "Move Player files to installation directory" ON)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    OPTION(BUNDLE_PYTHON "Builds CC3D assuming Python will be bundled with CC3D - default on Windows and OSX" ON)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    OPTION(BUNDLE_PYTHON "Builds CC3D assuming Python will be bundled with CC3D - default on Windows and OSX" ON)

else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    OPTION(BUNDLE_PYTHON "Builds CC3D assuming Python will be bundled with CC3D - default on Windows and OSX" OFF)
    SET(DEPENDENCIES_DIR CACHE PATH  "Location of dependencies folder. This folder will be copied to installation directory")
    OPTION(DEPENDENCIES_COPY_FLAG "Enables copying of the dependencies into installation dir" OFF)
endif()


# C++ only executable - useful for debugging
OPTION(BUILD_CPP_ONLY_EXECUTABLE "Build C++ only executable(requires Expat library)" OFF)

OPTION(BUILD_QT_WRAPPERS "Convenience CC3D/Player modules - e.g. redirecting text to Player console (requires Qt library and headers)" OFF )

#OPTION(BUILD_CUDA_MODULES "Enables building of CUDACC3D modules" OFF )
# OPTION(NO_CUDA "Disable automatic CUDA detection and building corresponding modules" OFF )
OPTION(NO_OPENCL "Disable automatic OpenCL detection and building corresponding modules" OFF )


OPTION(WINDOWS_DEPENDENCIES_INSTALL_ENABLE "Install Windows Dependencies - directory must be specified" ON)

if(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

    # set a path to VTK , PyQT and Qt libraries and to VS redeistributable package on windows
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows )
        SET(WINDOWS_DEPENDENCIES CACHE PATH  "Location of windows dependencies")
        LINK_DIRECTORIES(  ${WINDOWS_DEPENDENCIES}/lib)
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
endif(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)


include(compucell3d_cmake_macros.cmake)

# setting up compilation environment for XML parser based on Expat library
if(BUILD_CPP_ONLY_EXECUTABLE)

    SET(EXPAT_INCLUDE_DIRECTORY  CACHE PATH "EXPAT include directory")
    SET(EXPAT_LIBRARY_DIRECTORY  CACHE PATH "EXPAT library directory")

    SET(EXPAT_INCLUDE_DIR ${EXPAT_INCLUDE_DIRECTORY})
    SET(EXPAT_LIBRARY ${EXPAT_LIBRARY_DIRECTORY})

    INCLUDE(FindEXPAT)
    MESSAGE("EXPAT_INCLUDE_DIRS=" ${EXPAT_INCLUDE_DIRS})
    MESSAGE("EXPAT_LIBRARIES=" ${EXPAT_LIBRARIES})
    MESSAGE("EXPAT_FOUND=" ${EXPAT_FOUND})

    INCLUDE_DIRECTORIES (${EXPAT_INCLUDE_DIRS})
    LINK_DIRECTORIES(${EXPAT_LIBRARY})

endif(BUILD_CPP_ONLY_EXECUTABLE)

if (BUILD_QT_WRAPPERS)
    ADD_SUBDIRECTORY(QtWrappers)
endif (BUILD_QT_WRAPPERS)

ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(optimization)


