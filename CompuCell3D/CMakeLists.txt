CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
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
    # note: added -Wno-implicit-function-declaration -Wno-implicit-int to handle implicit function declarations in flex/bison generated code
    # this is stop gap measure and we need to either fix Units module or remove it since it is not used almost at all
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 -Wno-implicit-function-declaration -Wno-implicit-int" CACHE STRING "flags used during compilation by C++ compiler" FORCE)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -mmacosx-version-min=10.6 -O3 -g -fpermissive -m64 -Wno-implicit-function-declaration -Wno-implicit-int" CACHE STRING "flags used during compilation by C compiler" FORCE)
    # SET(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "OSX deployment target" FORCE)

    MESSAGE ("GOT APPLE")
ENDIF (APPLE)

set(CMAKE_CXX_STANDARD 11)

# to get openmp detected via conda - start cmake from the conda environment that contains
# openmp installation on OSX . on OSX install llvm-openmp to get openmp with clang
# When running via command line use e.g. :
# # os.system(f'source /Users/m/miniconda3/etc/profile.d/conda.sh ; conda activate cc3d_test ; {cmd_test}')
# if you have trouble after installation clear cache and remove cmake build directory - start fresh
FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message ("openmp c flags " "${OpenMP_C_FLAGS}")
    message ("openmp cxx flags " "${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    IF((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
    ENDIF()
ENDIF()


OPTION(FLEX_PYTHON_SPECS "Allow Flexible Python Path specification" OFF)
OPTION(ANACONDA_PYTHON_LAYOUT "Assume Python Distribution has similar directory layout as Python distribution in Anaconda" ON)


find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    get_filename_component(PYTHON_BASE_DIR "${Python3_EXECUTABLE}" DIRECTORY)
else()
    get_filename_component(PYTHON_BASE_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(PYTHON_BASE_DIR "${PYTHON_BASE_DIR}" DIRECTORY)
endif()

message("Python3_FOUND: ${Python3_FOUND}")
message("Python3_Interpreter_FOUND: ${Python3_Interpreter_FOUND}")
message("Python3_VERSION: ${Python3_VERSION}")
message("Python3_Development_FOUND: ${Python3_Development_FOUND}")
message("Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

message("Python3_Development_FOUND: ${Python3_Development_FOUND}")
message("Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("Python3_LIBRARIES: ${Python3_LIBRARIES}")
message("Python3_LIBRARY_RELEASE: ${Python3_LIBRARY_RELEASE}")
message("Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
message("Python3_RUNTIME_LIBRARY_DIRS: ${Python3_RUNTIME_LIBRARY_DIRS}")
message("Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")

message("Python3_LIBRARY_DIRS ${Python3_LIBRARY_DIRS}")
message("Python3_SABI_LIBRARY_DIRS ${Python3_SABI_LIBRARY_DIRS}")
message("Python3_SITEARCH ${Python3_SITEARCH}")

message("PYTHON_BASE_DIR${PYTHON_BASE_DIR}")

# Should we build shared libraries
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

SET(CC3D_PYTHON_MODULE_DIR ${CMAKE_SOURCE_DIR}/../)

# setting up version

set(COMPUCELL3D_MAJOR_VERSION 4)
set(COMPUCELL3D_MINOR_VERSION 6)
set(COMPUCELL3D_BUILD_VERSION 0)
set(COMPUCELL3D_REV_NUMBER 20240525)

## Specify a version for all libs to use
# letters A, B, C are added to ensure nice display of version in cmake gui
SET (COMPUCELL3D_A_MAJOR_VERSION ${COMPUCELL3D_MAJOR_VERSION} CACHE STRING "Major Version")
SET (COMPUCELL3D_B_MINOR_VERSION ${COMPUCELL3D_MINOR_VERSION} CACHE STRING "Minor Version")
SET (COMPUCELL3D_C_BUILD_VERSION ${COMPUCELL3D_BUILD_VERSION} CACHE STRING "Build Version")

# and here we assign them back to what they should be
SET (COMPUCELL3D_MAJOR_VERSION ${COMPUCELL3D_A_MAJOR_VERSION})
SET (COMPUCELL3D_MINOR_VERSION ${COMPUCELL3D_B_MINOR_VERSION})
SET (COMPUCELL3D_BUILD_VERSION ${COMPUCELL3D_C_BUILD_VERSION})

message(" THIS IS COMPUCELL3D_BUILD_VERSION " ${COMPUCELL3D_BUILD_VERSION})
message("COMPUCELL3D_C_BUILD_VERSION is " ${COMPUCELL3D_C_BUILD_VERSION})

SET(COMPUCELL3D_VERSION "${COMPUCELL3D_MAJOR_VERSION}.${COMPUCELL3D_MINOR_VERSION}.${COMPUCELL3D_BUILD_VERSION}")

message("GOT VERSION AS ${COMPUCELL3D_VERSION}")

# Set internal platform def
if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    add_definitions(-DCC3D_ISWIN)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
    add_definitions(-DCC3D_ISMAC)
endif()

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
OPTION(BUILD_STANDALONE "Build stand-alone CC3D installation" ON)

if(BUILD_STANDALONE)
    set(SITE_PACKAGES_INSTALL_REL lib/site-packages)
    set(COMPUCELL3D_INSTALL_SCRIPT_DIR_FULL_PATH ${CMAKE_INSTALL_PREFIX})
    set(COMPUCELL3D_INSTALL_SCRIPT_DIR .)
    SET(COMPUCELL3D_RUNTIME_DESTINATION bin)
    SET(COMPUCELL3D_LIBRARY_DESTINATION lib)
    SET(COMPUCELL3D_ARCHIVE_DESTINATION lib)
else()
    file(RELATIVE_PATH SITE_PACKAGES_INSTALL_REL ${CMAKE_INSTALL_PREFIX} ${Python3_SITELIB})
    set(COMPUCELL3D_INSTALL_SCRIPT_DIR_FULL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
    file(RELATIVE_PATH COMPUCELL3D_INSTALL_SCRIPT_DIR ${CMAKE_INSTALL_PREFIX} ${COMPUCELL3D_INSTALL_SCRIPT_DIR_FULL_PATH})


    if(ANACONDA_PYTHON_LAYOUT)
        # note this only works with same layouts as used in Anaconda.
        # if this is not the case set -DANACONDA_PYTHON_LAYOUT:BOOLEAN=OFF when calling cmake
        if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
            # If Python has  we install core shared libraries ((e.g. libCC3DBoundary) into appropriate location for shared library inside python distribution that was used to build CC3D
            SET(COMPUCELL3D_RUNTIME_DESTINATION ${PYTHON_BASE_DIR}/Library/bin)
            SET(COMPUCELL3D_LIBRARY_DESTINATION ${PYTHON_BASE_DIR}/Library/bin)
            SET(COMPUCELL3D_ARCHIVE_DESTINATION ${PYTHON_BASE_DIR}/Library/bin)
        else()
            # If Python is scattered all over the place we install core shared libraries (e.g. libCC3DBoundary) into CMAKE_INSTALL_PREFIX/bin
            SET(COMPUCELL3D_RUNTIME_DESTINATION ${PYTHON_BASE_DIR}/lib)
            SET(COMPUCELL3D_LIBRARY_DESTINATION ${PYTHON_BASE_DIR}/lib)
            SET(COMPUCELL3D_ARCHIVE_DESTINATION ${PYTHON_BASE_DIR}/lib)

        endif()
    else()
        SET(COMPUCELL3D_RUNTIME_DESTINATION bin)
        SET(COMPUCELL3D_LIBRARY_DESTINATION lib)
        SET(COMPUCELL3D_ARCHIVE_DESTINATION lib)

    endif()


endif()

SET(SITE_PACKAGES_INSTALL ${CMAKE_INSTALL_PREFIX}/${SITE_PACKAGES_INSTALL_REL})
SET(COMPUCELL3D_INSTALL_PY_MODULE_DIR ${SITE_PACKAGES_INSTALL_REL}/cc3d)
SET(COMPUCELL3D_INSTALL_SWIG_MODULES_DIR ${COMPUCELL3D_INSTALL_PY_MODULE_DIR}/cpp)

SET(COMPUCELL3D_INSTALL_BIN_DIR ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/bin)
SET(COMPUCELL3D_INSTALL_LIB_DIR ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/lib)
SET(COMPUCELL3D_INSTALL_PLUGIN_DIR ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/CompuCell3DPlugins)
SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/CompuCell3DSteppables)
SET(COMPUCELL3D_INSTALL_INCLUDE_DIR include/CompuCell3D)
SET(COMPUCELL3D_INSTALL_CONFIG_DIR lib/cmake/CompuCell3D)

# Find libz for use in the CompuCell/IO utils
INCLUDE(FindZLIB)

# Should we build wrappings
OPTION(BUILD_PYINTERFACE "Build Python language wrappings" ON)

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


if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    OPTION(WINDOWS_DEPENDENCIES_INSTALL_ENABLE "Install Windows Dependencies - directory must be specified" ON)
endif()

if(NOT ${BUILD_STANDALONE})
    set(BUNDLE_PYTHON OFF)
    set(WINDOWS_DEPENDENCIES_INSTALL_ENABLE OFF)
    set(DEPENDENCIES_COPY_FLAG OFF)
endif()


if(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

    # set a path to VTK , PyQT and Qt libraries and to VS redeistributable package on windows
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows )
        SET(WINDOWS_DEPENDENCIES CACHE PATH  "Location of windows dependencies")
        LINK_DIRECTORIES(  ${WINDOWS_DEPENDENCIES}/lib)
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
endif(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

# handle rpath
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if(APPLE)
    set(RPATH_BASE @loader_path)
else()
    set(RPATH_BASE $ORIGIN)
endif()
file(RELATIVE_PATH BinToLibRelPath
    ${CMAKE_INSTALL_PREFIX}/bin
    ${CMAKE_INSTALL_PREFIX}/lib
)
file(RELATIVE_PATH CPPLibToLibRelPath
    ${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}
    ${CMAKE_INSTALL_PREFIX}/lib
)
set(CMAKE_INSTALL_RPATH ${RPATH_BASE} ${RPATH_BASE}/${BinToLibRelPath} ${RPATH_BASE}/${CPPLibToLibRelPath} ${RPATH_BASE}/CompuCell3DPlugins ${RPATH_BASE}/CompuCell3DSteppables)


include(compucell3d_cmake_macros.cmake)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

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
# ADD_SUBDIRECTORY(optimization)

# install info files
install(FILES
    License.txt
    ReleaseNotes.rst
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}
)
