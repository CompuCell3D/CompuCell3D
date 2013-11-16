CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(COMPUCELL3D)

FIND_PACKAGE (OpenMP)    
# Should we build shared libraries
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Specify a version for all libs to use
# Specify a version for all libs to use
# letters A, B, C are added to ensure nice display of sersion in cmake gui
SET (COMPUCELL3D_A_MAJOR_VERSION "3" CACHE STRING "Major Version")
SET (COMPUCELL3D_B_MINOR_VERSION "7" CACHE STRING "Minor Version")
SET (COMPUCELL3D_C_BUILD_VERSION "1" CACHE STRING "Build Version")
# and here we assign them back to what they shuld be
SET (COMPUCELL3D_MAJOR_VERSION ${COMPUCELL3D_A_MAJOR_VERSION})
SET (COMPUCELL3D_MINOR_VERSION ${COMPUCELL3D_B_MINOR_VERSION})
SET (COMPUCELL3D_BUILD_VERSION ${COMPUCELL3D_C_BUILD_VERSION})

SET(COMPUCELL3D_LIBRARY_PROPERTIES ${COMPUCELL3D_LIBRARY_PROPERTIES}
  VERSION "${COMPUCELL3D_VERSION}"
  SOVERSION "${COMPUCELL3D_MAJOR_VERSION}.${COMPUCELL3D_MINOR_VERSION}" )


SET(COMPUCELL3D_INSTALL_PATH CACHE PATH "CompuCell3D installation path")
SET(COMPUCELL3D_FULL_SOURCE_PATH CACHE PATH "CompuCell3D source code path ")

SET(COMPUCELL3D_INCLUDE_PATH ${COMPUCELL3D_INSTALL_PATH}/include/CompuCell3D)
SET(COMPUCELL3D_LIBRARY_PATH ${COMPUCELL3D_INSTALL_PATH}/lib)


# Xerces-c dependency
# FIND_PATH(XERCESC_INCLUDE_DIR xercesc/util/XercesDefs.hpp
  # /usr/include
  # /usr/local/include
  # /opt/local/include
  # ${COMPUCELL3D_FULL_SOURCE_PATH}/xercesc/include
  # )

# FIND_LIBRARY(XERCESC_LIBRARY xerces-c
  # /usr/lib
  # /usr/local/lib
  # /opt/local/lib
  # ${COMPUCELL3D_FULL_SOURCE_PATH}/xercesc/lib

  # )


# SET(COMPUCELL3D_INCLUDE_PATH CACHE PATH "Path To CompuCell3D Include Files")
# SET(COMPUCELL3D_LIBRARY_PATH CACHE PATH "Path To CompuCell3D Libraries")


# SET(CC3DLibraries ${COMPUCELL3D_LIBRARY_PATH}/CC3DCompuCellLib ${COMPUCELL3D_LIBRARY_PATH}/CC3DBoundary ${COMPUCELL3D_LIBRARY_PATH}/CC3DBasicUtils ${COMPUCELL3D_LIBRARY_PATH}/CC3DField3D ${COMPUCELL3D_LIBRARY_PATH}/CC3DPublicUtilities)

LINK_DIRECTORIES(${COMPUCELL3D_LIBRARY_PATH})
SET(CC3DLibraries CC3DCompuCellLib CC3DBoundary CC3DBasicUtils CC3DField3D CC3DPublicUtilities CC3DUnits CC3DXMLUtils CC3DmuParser)


MESSAGE("CC3DLibraries=${CC3DLibraries}")

# SET(CompuCellLibShared CACHE FILEPATH "CompuCell3DLibrary")
# 
# SET(BasicUtilsLib CACHE FILEPATH "BasicUtilsLib")
# 
# SET(Field3DLib CACHE FILEPATH "Field3DLib")
# SET(BoundaryLib CACHE FILEPATH "BoundaryLib")

MESSAGE("THIS IS COMPUCELL3D_INCLUDE_PATH=${COMPUCELL3D_INCLUDE_PATH}")
# Always include the top level source for compiles
SET_DIRECTORY_PROPERTIES(PROPERTIES
  INCLUDE_DIRECTORIES ${COMPUCELL3D_INCLUDE_PATH})

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

  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})

  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DPlugins RUNTIME_DIRECTORY /lib/CompuCell3DPlugins
  ${LIBRARY_NAME}Shared)

ENDMACRO(ADD_COMPUCELL3D_PLUGIN)

MACRO(ADD_COMPUCELL3D_PLUGIN_HEADERS plugin)

  file(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  # message(${plugin} "uses the following header files: " ${header_files})
  file(INSTALL DESTINATION "@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/plugins/${plugin}" TYPE FILE FILES ${header_files})

  # INSTALL_FILES(/include/CompuCell3D/CompuCell3D/plugins/${plugin} .h
    # ${ARGN})
ENDMACRO(ADD_COMPUCELL3D_PLUGIN_HEADERS)

MACRO(ADD_COMPUCELL3D_STEPPABLE_HEADERS steppable)
  file(GLOB header_files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  # message(${plugin} "uses the following header files: " ${header_files})
  file(INSTALL DESTINATION "@CMAKE_INSTALL_PREFIX@/include/CompuCell3D/CompuCell3D/steppables/${steppable}" TYPE FILE FILES ${header_files})

  # INSTALL_FILES(/include/CompuCell3D/CompuCell3D/steppables/${steppable} .h
    # ${ARGN})
ENDMACRO(ADD_COMPUCELL3D_STEPPABLE_HEADERS)

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

  ADD_LIBRARY(${LIBRARY_NAME}Shared SHARED ${LIBRARY_SOURCES})
  TARGET_LINK_LIBRARIES(${LIBRARY_NAME}Shared ${LIBRARY_LINK_LIBRARIES})
  SET_TARGET_PROPERTIES(
  ${LIBRARY_NAME}Shared PROPERTIES
  ${COMPUCELL3D_LIBRARY_PROPERTIES})

  SET_TARGET_PROPERTIES(${LIBRARY_NAME}Shared  PROPERTIES OUTPUT_NAME CC3D${LIBRARY_NAME}${LIBRARY_SUFFIX} COMPILE_FLAGS "${LIBRARY_EXTRA_COMPILER_FLAGS}")
  INSTALL_TARGETS(/lib/CompuCell3DSteppables RUNTIME_DIRECTORY /lib/CompuCell3DSteppables
  ${LIBRARY_NAME}Shared)

ENDMACRO(ADD_COMPUCELL3D_STEPPABLE)




# ADD YOUR SUBPROJECTS HERE
#AutogeneratedModules - DO NOT REMOVE THIS LINE IT IS USED BY TWEDIT TO INSERT NEW PLUGIN DIRECTORY

add_subdirectory(SimpleVolume)
add_subdirectory(VolumeMean)
add_subdirectory(pyinterface)
