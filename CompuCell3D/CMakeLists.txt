CMAKE_MINIMUM_REQUIRED(VERSION 2.6OD)
PROJECT(COMPUCELL3D)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

if (${CMAKE_VERSION} VERSION_GREATER 3.1.0)
    cmake_policy(SET CMP0053 OLD)
endif()


#APPLE CHECK
IF (APPLE)
#	SET (CMAKE_OSX_ARCHITECTURES "i386")            #x86_64 ppc ppc64
#	SET (CMAKE_OSX_SYSROOT "/Developer/SDKs/MacOSX10.5.sdk")
#	SET (CMAKE_CXX_FLAGS -mmacosx-version-min=10.4)
#	SET (CMAKE_C_FLAGS -mmacosx-version-min=10.4)
	MESSAGE ("GOT APPLE")
ENDIF (APPLE)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET (CMAKE_CXX_FLAGS -std=c++0x)
endif()

FIND_PACKAGE (OpenMP)

# Should we build shared libraries
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

SET (PYQT_VERSION "5" CACHE STRING "PyQt major version")

# Specify a version for all libs to use
# letters A, B, C are added to ensure nice display of version in cmake gui
SET (COMPUCELL3D_A_MAJOR_VERSION "3" CACHE STRING "Major Version")
SET (COMPUCELL3D_B_MINOR_VERSION "7" CACHE STRING "Minor Version")
SET (COMPUCELL3D_C_BUILD_VERSION "9" CACHE STRING "Build Version")


# and here we assign them back to what they shuld be
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


INCLUDE_DIRECTORIES (
    ${COMPUCELL3D_SOURCE_DIR}/core
    )


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

#if(COMMAND cmake_policy)
#  cmake_policy(SET CMP0002 NEW)
#endif(COMMAND cmake_policy)

# Always include the top level source for compiles
SET_DIRECTORY_PROPERTIES(PROPERTIES
  INCLUDE_DIRECTORIES ${COMPUCELL3D_SOURCE_DIR})

# Where to install files relative to prefix
SET(COMPUCELL3D_INSTALL_LIB_DIR "/lib")
SET(COMPUCELL3D_INSTALL_PLUGIN_DIR "/lib/CompuCell3DPlugins")
SET(COMPUCELL3D_INSTALL_STEPPABLE_DIR "/lib/CompuCell3DSteppables")
SET(COMPUCELL3D_INSTALL_INCLUDE_DIR "/include/CompuCell3D")

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

# if (EXISTS ${DEPENDENCIES_DIR})
#     message("depandencies exist")
# else()
#     message("depandencies missing")
# endif()







# C++ only executable - useful for debugging
OPTION(BUILD_CPP_ONLY_EXECUTABLE "Build C++ only executable(requires Expat library)" OFF)

OPTION(BUILD_QT_WRAPPERS "Convenience CC3D/Player modules - e.g. redirecting text to Player console (requires Qt library and headers)" OFF )

#OPTION(BUILD_CUDA_MODULES "Enables building of CUDACC3D modules" OFF )
OPTION(NO_OPENCL "Disable automatic OpenCL detection and building corresponding modules" OFF )
#OPTION(NO_CUDA "Disable automatic CUDA detection and building corresponding modules" OFF )

OPTION(WINDOWS_DEPENDENCIES_INSTALL_ENABLE "Install Windows Dependencies - directory must be specified" ON)

if(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

    # set a path to VTK , PyQT and Qt libraries and to VS redeistributable package on windows
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows )
        SET(WINDOWS_DEPENDENCIES CACHE PATH  "Location of windows dependencies")
        LINK_DIRECTORIES(  ${WINDOWS_DEPENDENCIES}/lib)
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
endif(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

SET(COMPUCELL3D_PLUGIN_PATH
    /lib/CompuCell3DPlugins)
#ENDIF(NOT COMPUCELL3D_PLUGIN_PATH)


SET(COMPUCELL3D_STEPPABLE_PATH
    /lib/CompuCell3DSteppables)

#helpful macros
MACRO(LIST_CONTAINS var value)
  SET(${var})
  FOREACH (value2 ${ARGN})
    IF (${value} STREQUAL ${value2})
      SET(${var} TRUE)
    ENDIF (${value} STREQUAL ${value2})
  ENDFOREACH (value2)
ENDMACRO(LIST_CONTAINS)

MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name})
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} FALSE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    LIST_CONTAINS(is_arg_name ${arg} ${arg_names})
    IF (is_arg_name)
      SET(${prefix}_${current_arg_name} ${current_arg_list})
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSE (is_arg_name)
      LIST_CONTAINS(is_option ${arg} ${option_names})
      IF (is_option)
	SET(${prefix}_${arg} TRUE)
      ELSE (is_option)
	SET(current_arg_list ${current_arg_list} ${arg})
      ENDIF (is_option)
    ENDIF (is_arg_name)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)


MACRO(CAR var)
  SET(${var} ${ARGV1})
ENDMACRO(CAR)


MACRO(CDR var junk)
  SET(${var} ${ARGN})
ENDMACRO(CDR)

# notice that variables listed in PARSE_ARGUMENTS function (e.g. "LINK_LIBRARIES;DEPENDS;SUFFIX;COMPILE_FLAGS") have to be prepended with "LIBRARY_" before being referenced
MACRO(ADD_STATIC_LIBRARY)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  # MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  # MESSAGE("STATIC LIBLink libraries: ${LIBRARY_LINK_LIBRARIES}")
  # MESSAGE("STATIC LIB SUFFIX: ${LIBRARY_SUFFIX}")
  # MESSAGE("STATIC LIB COMPILE_FLAGS: ${LIBRARY_EXTRA_COMPILER_FLAGS}")

  ADD_LIBRARY(${LIBRARY_NAME}Static STATIC ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Static ${LIBRARY_LINK_LIBRARIES})

    IF(USE_LIBRARY_VERSIONS)
      SET_TARGET_PROPERTIES(
      ${LIBRARY_NAME}Static PROPERTIES
      ${COMPUCELL3D_LIBRARY_PROPERTIES})
    ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Static  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib ${LIBRARY_NAME}Static)
ENDMACRO(ADD_STATIC_LIBRARY)

# notice that variables listed in PARSE_ARGUMENTS function (e.g. "LINK_LIBRARIES;DEPENDS;SUFFIX;COMPILE_FLAGS") have to be prepended with "LIBRARY_" before being referenced
MACRO(ADD_SHARED_LIBRARY)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  #MESSAGE("Link libraries: ${LIBRARY_LINK_LIBRARIES}")

  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
IF(USE_LIBRARY_VERSIONS)
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib ${LIBRARY_NAME}Shared)
ENDMACRO(ADD_SHARED_LIBRARY)

MACRO(ADD_HEADERS targetDir)
  # old style command
  # INSTALL_FILES(/include/CompuCell3D/CompuCell3D/plugins/${plugin} .h
    # ${ARGN})

  file(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx")
  # message(${plugin} "uses the following header files: " ${header_files})
  # INSTALL(FILES ${header_files} DESTINATION "@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/plugins/${plugin}")
  INSTALL(FILES ${header_files} DESTINATION ${targetDir})

ENDMACRO(ADD_HEADERS)

MACRO(ADD_COMPUCELL3D_PLUGIN_HEADERS plugin)
  # old style command
  # INSTALL_FILES(/include/CompuCell3D/CompuCell3D/plugins/${plugin} .h
    # ${ARGN})

  file(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  # message(${plugin} "uses the following header files: " ${header_files})
  INSTALL(FILES ${header_files} DESTINATION "@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/plugins/${plugin}")


ENDMACRO(ADD_COMPUCELL3D_PLUGIN_HEADERS)

MACRO(ADD_COMPUCELL3D_PLUGIN)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  #MESSAGE("Link libraries: ${LIBRARY_LINK_LIBRARIES}")
  # ADD_DEFINITIONS(-DEXP_STL)
  # # # #Automatically write a dll header file - used once during code rewriting
  # # # set(LIBRARY_NAME_UPPERCASE ${LIBRARY_NAME})
  # # # #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  # # # #for more info please see http://www.amk.ca/python/howto/regex/
  # # # STRING(TOUPPER ${LIBRARY_NAME_UPPERCASE} LIBRARY_NAME_UPPERCASE )

  # # # configure_file(${CMAKE_SOURCE_DIR}/core/DLLSpecifier.h.in ${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_NAME}DLLSpecifier.h @ONLY)
  # # # message("current_src_dir" ${CMAKE_SOURCE_DIR})
  # # # message("library_name" ${LIBRARY_NAME}DLLSpecifier.h)

  # message("library_name " ${LIBRARY_NAME})

  INCLUDE_DIRECTORIES (
        ${COMPUCELL3D_SOURCE_DIR}/core
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/plugins
  )

  file(GLOB source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx" "${CMAKE_CURRENT_SOURCE_DIR}/*.c" )
  # message(${plugin} "uses the following header files: " ${header_files})
  # message("sources " ${source_files})



  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES} ${source_files})
  # ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
IF(USE_LIBRARY_VERSIONS)
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DPlugins RUNTIME_DIRECTORY /lib/CompuCell3DPlugins
  ${LIBRARY_NAME}Shared)

  # installing headers

  # file(GLOB _HEADERS *.h)
  # list(APPEND HEADERS ${_HEADERS})
  # install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)
  # ADD_COMPUCELL3D_PLUGIN_HEADERS(${LIBRARY_NAME})
  ADD_HEADERS("@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/plugins/${LIBRARY_NAME}")

ENDMACRO(ADD_COMPUCELL3D_PLUGIN)


MACRO(ADD_COMPUCELL3D_PLUGIN_ORIGINAL)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  #MESSAGE("Link libraries: ${LIBRARY_LINK_LIBRARIES}")
  # ADD_DEFINITIONS(-DEXP_STL)
  # # # #Automatically write a dll header file - used once during code rewriting
  # # # set(LIBRARY_NAME_UPPERCASE ${LIBRARY_NAME})
  # # # #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  # # # #for more info please see http://www.amk.ca/python/howto/regex/
  # # # STRING(TOUPPER ${LIBRARY_NAME_UPPERCASE} LIBRARY_NAME_UPPERCASE )

  # # # configure_file(${CMAKE_SOURCE_DIR}/core/DLLSpecifier.h.in ${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_NAME}DLLSpecifier.h @ONLY)
  # # # message("current_src_dir" ${CMAKE_SOURCE_DIR})
  # # # message("library_name" ${LIBRARY_NAME}DLLSpecifier.h)

  # message("library_name " ${LIBRARY_NAME})
  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
IF(USE_LIBRARY_VERSIONS)
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DPlugins RUNTIME_DIRECTORY /lib/CompuCell3DPlugins
  ${LIBRARY_NAME}Shared)

  # installing headers

  # file(GLOB _HEADERS *.h)
  # list(APPEND HEADERS ${_HEADERS})
  # install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)
  ADD_COMPUCELL3D_PLUGIN_HEADERS(${LIBRARY_NAME})

ENDMACRO(ADD_COMPUCELL3D_PLUGIN_ORIGINAL)


MACRO(ADD_COMPUCELL3D_STEPPABLE)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  #MESSAGE("Link libraries: ${LIBRARY_LINK_LIBRARIES}")

  # # # #Automatically write a dll header file - used once during code rewriting
  # # # set(LIBRARY_NAME_UPPERCASE ${LIBRARY_NAME})
  # # # #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  # # # #for more info please see http://www.amk.ca/python/howto/regex/
  # # # STRING(TOUPPER ${LIBRARY_NAME_UPPERCASE} LIBRARY_NAME_UPPERCASE )

  # # # configure_file(${CMAKE_SOURCE_DIR}/core/DLLSpecifier.h.in ${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_NAME}DLLSpecifier.h @ONLY)
  # # # message("current_src_dir" ${CMAKE_SOURCE_DIR})
  # # # message("library_name" ${LIBRARY_NAME}DLLSpecifier.h)

  INCLUDE_DIRECTORIES (
        ${COMPUCELL3D_SOURCE_DIR}/core
        ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D/steppables
  )

  file(GLOB source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx" "${CMAKE_CURRENT_SOURCE_DIR}/*.c" )

  # message("sources " ${source_files})
  # message (LIBRARY_SOURCES ${LIBRARY_SOURCES})

  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES} ${source_files} )
  # ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})

IF(USE_LIBRARY_VERSIONS)
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DSteppables RUNTIME_DIRECTORY /lib/CompuCell3DSteppables ${LIBRARY_NAME}Shared)

  ADD_HEADERS("@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/steppables/${LIBRARY_NAME}")

ENDMACRO(ADD_COMPUCELL3D_STEPPABLE)

MACRO(ADD_COMPUCELL3D_STEPPABLE_ORIGINAL)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")
  #MESSAGE("Link libraries: ${LIBRARY_LINK_LIBRARIES}")

  # # # #Automatically write a dll header file - used once during code rewriting
  # # # set(LIBRARY_NAME_UPPERCASE ${LIBRARY_NAME})
  # # # #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  # # # #for more info please see http://www.amk.ca/python/howto/regex/
  # # # STRING(TOUPPER ${LIBRARY_NAME_UPPERCASE} LIBRARY_NAME_UPPERCASE )

  # # # configure_file(${CMAKE_SOURCE_DIR}/core/DLLSpecifier.h.in ${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_NAME}DLLSpecifier.h @ONLY)
  # # # message("current_src_dir" ${CMAKE_SOURCE_DIR})
  # # # message("library_name" ${LIBRARY_NAME}DLLSpecifier.h)


  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
IF(USE_LIBRARY_VERSIONS)
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})
ENDIF(USE_LIBRARY_VERSIONS)

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DSteppables RUNTIME_DIRECTORY /lib/CompuCell3DSteppables
  ${LIBRARY_NAME}Shared)
ENDMACRO(ADD_COMPUCELL3D_STEPPABLE_ORIGINAL)


# setting up compilation environment for XML parser based on Expat library
if(BUILD_CPP_ONLY_EXECUTABLE)


    SET(EXPAT_INCLUDE_DIRECTORY  CACHE PATH "EXPAT include directory")
    SET(EXPAT_LIBRARY_DIRECTORY  CACHE PATH "EXPAT library directory")

    # MESSAGE("EXPAT_INCLUDE_DIR=",${EXPAT_INCLUDE_DIR})
    # MESSAGE("EXPAT_INCLUDE_DIRECTORY=",${EXPAT_INCLUDE_DIRECTORY})

    SET(EXPAT_INCLUDE_DIR ${EXPAT_INCLUDE_DIRECTORY})
    SET(EXPAT_LIBRARY ${EXPAT_LIBRARY_DIRECTORY})


    INCLUDE(FindEXPAT)
    MESSAGE("EXPAT_INCLUDE_DIRS=" ${EXPAT_INCLUDE_DIRS})
    MESSAGE("EXPAT_LIBRARIES=" ${EXPAT_LIBRARIES})
    MESSAGE("EXPAT_FOUND=" ${EXPAT_FOUND})


    INCLUDE_DIRECTORIES (
        ${EXPAT_INCLUDE_DIRS}
        )
    LINK_DIRECTORIES(${EXPAT_LIBRARY})

endif(BUILD_CPP_ONLY_EXECUTABLE)

if (BUILD_QT_WRAPPERS)
    ADD_SUBDIRECTORY(QtWrappers)
endif (BUILD_QT_WRAPPERS)


ADD_SUBDIRECTORY(core)

ADD_SUBDIRECTORY(optimization)

if(${PYQT_VERSION} STREQUAL 4)
    ADD_SUBDIRECTORY(player)
    ADD_SUBDIRECTORY(Twedit++)
elseif(${PYQT_VERSION} STREQUAL 5)
    ADD_SUBDIRECTORY(player5)
    ADD_SUBDIRECTORY(Twedit++5)
endif()
