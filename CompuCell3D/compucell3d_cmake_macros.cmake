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

# notice that variables listed in PARSE_ARGUMENTS function (e.g. "LINK_LIBRARIES;DEPENDS;SUFFIX;COMPILE_FLAGS")
# have to be prepended with "LIBRARY_" before being referenced
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
  # INSTALL_TARGETS(/lib ${LIBRARY_NAME}Shared)
  INSTALL_TARGETS( /lib/site-packages/cc3d/cpp/lib  RUNTIME_DIRECTORY /lib/site-packages/cc3d/cpp/bin ${LIBRARY_NAME}Shared)
#  RUNTIME_DIRECTORY "${SITE_PACKAGES_INSTALL}/cc3d/cpp/bin"
# INSTALL_TARGETS( "${COMPUCELL3D_INSTALL_PLUGIN_DIR}" RUNTIME_DIRECTORY "${COMPUCELL3D_INSTALL_PLUGIN_DIR}" ${LIBRARY_NAME}Shared )



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
  INSTALL(FILES ${header_files} DESTINATION "${CMAKE_INSTALL_PREFIX}/include/CompuCell3D/CompuCell3D/plugins/${plugin}")


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
  INSTALL_TARGETS( "${COMPUCELL3D_INSTALL_PLUGIN_DIR}" RUNTIME_DIRECTORY "${COMPUCELL3D_INSTALL_PLUGIN_DIR}" ${LIBRARY_NAME}Shared )

  # installing headers

  # file(GLOB _HEADERS *.h)
  # list(APPEND HEADERS ${_HEADERS})
  # install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)
  # ADD_COMPUCELL3D_PLUGIN_HEADERS(${LIBRARY_NAME})
  ADD_HEADERS("${CMAKE_INSTALL_PREFIX}/include/CompuCell3D/CompuCell3D/plugins/${LIBRARY_NAME}")

ENDMACRO(ADD_COMPUCELL3D_PLUGIN)




MACRO(ADD_COMPUCELL3D_STEPPABLE)
   PARSE_ARGUMENTS(LIBRARY
    "LINK_LIBRARIES;DEPENDS;SUFFIX;EXTRA_COMPILER_FLAGS"
    ""
    ${ARGN}
    )
  CAR(LIBRARY_NAME ${LIBRARY_DEFAULT_ARGS})
  CDR(LIBRARY_SOURCES ${LIBRARY_DEFAULT_ARGS})

  #MESSAGE("*** NAME OF LIBRARY ${LIBRARY_NAME}")
  #MESSAGE("Sources: ${LIBRARY_SOURCES}")`
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
  INSTALL_TARGETS(${COMPUCELL3D_INSTALL_STEPPABLE_DIR} RUNTIME_DIRECTORY ${COMPUCELL3D_INSTALL_STEPPABLE_DIR} ${LIBRARY_NAME}Shared)

  ADD_HEADERS("${CMAKE_INSTALL_PREFIX}/include/CompuCell3D/CompuCell3D/steppables/${LIBRARY_NAME}")

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
  INSTALL_TARGETS(${COMPUCELL3D_INSTALL_STEPPABLE_DIR} RUNTIME_DIRECTORY ${COMPUCELL3D_INSTALL_STEPPABLE_DIR} ${LIBRARY_NAME}Shared)
ENDMACRO(ADD_COMPUCELL3D_STEPPABLE_ORIGINAL)


