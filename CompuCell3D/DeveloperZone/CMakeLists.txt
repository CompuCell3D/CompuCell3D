CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(DeveloperZone)


# Should we build shared libraries
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

IF (APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 " CACHE STRING "flags used during compilation by C++ compiler" FORCE)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 " CACHE STRING "flags used during compilation by C compiler" FORCE)
    SET(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "OSX deployment target" FORCE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        # M1, arm64 processor
        SET(CMAKE_OSX_SYSROOT /opt/MacOSX11.3.sdk)
    else()
        #x86 Intel machines
        SET(CMAKE_OSX_SYSROOT /opt/MacOSX10.10.sdk)
    endif()
    MESSAGE ("GOT APPLE")
ENDIF (APPLE)

# Set internal platform def
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    add_definitions(-DCC3D_ISWIN)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    add_definitions(-DCC3D_ISMAC)
endif()

SET(CMAKE_CXX_STANDARD 11)
# to get openmp detected via conda - start cmake from the conda environment that contains
# openmp installation on OSX

find_library(OpenMP_LIBRARY
    NAMES omp
)

# find_path(OpenMP_INCLUDE_DIR
#     omp.h
# )

# mark_as_advanced(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)
#
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(OpenMP DEFAULT_MSG
#     OpenMP_LIBRARY
#     # OpenMP_INCLUDE_DIR
# )
#
# if (OpenMP_FOUND)
#     set(OpenMP_LIBRARIES ${OpenMP_LIBRARY})
#     # set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR})
#     set(OpenMP_COMPILE_OPTIONS -Xpreprocessor -fopenmp)
#
#     add_library(OpenMP::OpenMP SHARED IMPORTED)
#     set_target_properties(OpenMP::OpenMP PROPERTIES
#         IMPORTED_LOCATION ${OpenMP_LIBRARIES}
#         INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
#         INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
#     )
# endif()




FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    message ("openmp c flags " "${OpenMP_C_FLAGS}")
    message ("openmp cxx flags " "${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    IF((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
    ENDIF()
ENDIF()


# Specify a version for all libs to use
# Specify a version for all libs to use
# letters A, B, C are added to ensure nice display of sersion in cmake gui
SET (COMPUCELL3D_A_MAJOR_VERSION "4" CACHE STRING "Major Version")
SET (COMPUCELL3D_B_MINOR_VERSION "0" CACHE STRING "Minor Version")
SET (COMPUCELL3D_C_BUILD_VERSION "0" CACHE STRING "Build Version")
# and here we assign them back to what they shuld be
SET (COMPUCELL3D_MAJOR_VERSION ${COMPUCELL3D_A_MAJOR_VERSION})
SET (COMPUCELL3D_MINOR_VERSION ${COMPUCELL3D_B_MINOR_VERSION})
SET (COMPUCELL3D_BUILD_VERSION ${COMPUCELL3D_C_BUILD_VERSION})

SET(COMPUCELL3D_LIBRARY_PROPERTIES ${COMPUCELL3D_LIBRARY_PROPERTIES}
  VERSION "${COMPUCELL3D_VERSION}"
  SOVERSION "${COMPUCELL3D_MAJOR_VERSION}.${COMPUCELL3D_MINOR_VERSION}" )


SET(COMPUCELL3D_INSTALL_PATH CACHE PATH "CompuCell3D installation path")
SET(COMPUCELL3D_GIT_DIR CACHE PATH "CompuCell3D Git Repository path ")

SET(COMPUCELL3D_FULL_SOURCE_PATH ${COMPUCELL3D_GIT_DIR}/CompuCell3D/core/CompuCell3D)



SET(COMPUCELL3D_FULL_SOURCE_PATH CACHE PATH "CompuCell3D source code path ")
SET(COMPUCELL3D_SOURCE_DIR CACHE PATH "CompuCell3D source code path <CC3D_GIT_REPO>/Comp ")
SET(COMPUCELL3D_SOURCE_DIR ${COMPUCELL3D_GIT_DIR}/CompuCell3D)

# assuming cmake_install_prefix points to site packages dir we do the following:
SET(COMPUCELL3D_INSTALL_LIB_DIR ${COMPUCELL3D_INSTALL_PATH}/cc3d/cpp/lib)

# for alternative (old cc3d layout we would do this)
#SET(COMPUCELL3D_INSTALL_LIB_DIR ${COMPUCELL3D_INSTALL_PATH}/lib/site-packages/cc3d/cpp/lib)

SET(COMPUCELL3D_INCLUDE_PATH ${COMPUCELL3D_INSTALL_PATH}/include/CompuCell3D)
SET(COMPUCELL3D_LIBRARY_PATH ${COMPUCELL3D_INSTALL_PATH}/lib)

SET(COMPUCELL3D_LINK_LIBRARY_PATH )
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # ${COMPUCELL3D_INSTALL_PATH} points to <conda_env>/Lib/site_packages
    # and link librariers .lib are in <conda_env>/Library/lib
    SET(COMPUCELL3D_LINK_LIBRARY_PATH ${COMPUCELL3D_INSTALL_PATH}/../../Library/lib)
elseif (UNIX)
    SET(COMPUCELL3D_LINK_LIBRARY_PATH ${COMPUCELL3D_INSTALL_PATH}/../../)
endif()


# assuming cmake_install_prexif points to site packages dir we do the following:
SET(COMPUCELL3D_INSTALL_PLUGIN_DIR " cc3d/cpp/CompuCell3DPlugins")
SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR " cc3d/cpp/CompuCell3DSteppables")

# for alternative (old cc3d layout we would do this)
# SET(COMPUCELL3D_INSTALL_PLUGIN_DIR " lib/site-packages/cc3d/cpp/CompuCell3DPlugins")
# SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR " lib/site-packages/cc3d/cpp/CompuCell3DSteppables")


LINK_DIRECTORIES(${COMPUCELL3D_LIBRARY_PATH} ${COMPUCELL3D_INSTALL_LIB_DIR} ${COMPUCELL3D_LINK_LIBRARY_PATH})
SET(CC3DLibraries
    CC3DLogger
    CC3DCompuCellLib
    CC3DAutomaton
    CC3DBoundary
    CC3DField3D
    CC3DPublicUtilities
    CC3DXMLUtils
    CC3DPotts3D
    CC3DmuParser
    CC3DExpressionEvaluator

    )
SET(STEPPABLE_DEPENDENCIES ${CC3DLibraries})
SET(PLUGIN_DEPENDENCIES ${CC3DLibraries})

MESSAGE("CC3DLibraries=${CC3DLibraries}")

MESSAGE("THIS IS COMPUCELL3D_INCLUDE_PATH=$ {COMPUCELL3D_INCLUDE_PATH}")
# Always include the top level source for compiles
SET_DIRECTORY_PROPERTIES(PROPERTIES
  INCLUDE_DIRECTORIES ${COMPUCELL3D_INCLUDE_PATH})

include(developer_zone_macros.cmake)


# ADD YOUR SUBPROJECTS HERE
#AutogeneratedModules - DO NOT REMOVE THIS LINE IT IS USED BY TWEDIT TO INSERT NEW PLUGIN DIRECTORY
ADD_SUBDIRECTORY(CustomCellAttributeSteppable)
ADD_SUBDIRECTORY(HeterotypicBoundaryLength)
ADD_SUBDIRECTORY(GrowthSteppable)


add_subdirectory(SimpleVolume)
add_subdirectory(VolumeMean)
add_subdirectory(pyinterface)

message("VAR COMPUCELL3D_SOURCE_DIR" ${COMPUCELL3D_SOURCE_DIR})
