SET(TOP_LEVEL_DIR ${CMAKE_SOURCE_DIR}/../)

#writing run scripts

set(LIBRARY_PATH_SYNTAX LD_LIBRARY_PATH) #default choice
if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
   set(LIBRARY_PATH_SYNTAX LD_LIBRARY_PATH)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
   set(LIBRARY_PATH_SYNTAX DYLD_LIBRARY_PATH)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

SET(COMPUCELL_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})

set(cc3d_py_dir cc3d)
set(cc3d_py_dir_target lib/site-packages/${cc3d_py_dir})
set(cc3d_py_source_dir ${CMAKE_SOURCE_DIR}/../cc3d)

message("THIS IS cc3d_py_source_dir" ${cc3d_py_source_dir})

if(INSTALL_PLAYER)
#INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/${cc3d_py_dir}/CompuCellSetup" DESTINATION ${cc3d_py_dir_target}
    INSTALL(FILES  ${cc3d_py_source_dir}/__init__.py DESTINATION ${cc3d_py_dir_target})
    INSTALL(FILES  ${cc3d_py_source_dir}/run_script.py DESTINATION ${cc3d_py_dir_target})

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/CompuCellSetup DESTINATION ${cc3d_py_dir_target}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/CompuCellSetup/*)

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/core DESTINATION ${cc3d_py_dir_target}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/core/*)

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/player5 DESTINATION ${cc3d_py_dir_target}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/player5/*)

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/twedit5 DESTINATION ${cc3d_py_dir_target}
           PATTERN "*.svn" EXCLUDE
           PATTERN "*.ui" EXCLUDE
           PATTERN "*.in" EXCLUDE
           PATTERN "*.pyc" EXCLUDE
           PATTERN "*.bak" EXCLUDE
           PATTERN "__pycache__" EXCLUDE
           PATTERN ${cc3d_py_source_dir}/twedit5/*)

endif()

if (${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    if(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)

       INSTALL(DIRECTORY  "${WINDOWS_DEPENDENCIES}/" DESTINATION ${CMAKE_INSTALL_PREFIX}
           PATTERN "*.svn" EXCLUDE
           PATTERN "${WINDOWS_DEPENDENCIES}/*")

    endif(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  #here I need to change "/" to "\" to comply with windows style paths
  set(COMPUCELL_INSTALL_DIR_WINDOWS_STYLE ${COMPUCELL_INSTALL_DIR})
  set(RUN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/windows)

  #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  #for more info please see http://www.amk.ca/python/howto/regex/
  STRING(REGEX REPLACE "/" "\\\\" COMPUCELL_INSTALL_DIR_WINDOWS_STYLE ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE})

  configure_file(${RUN_SCRIPT_DIR}/cc3d.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/cc3d.bat @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/run_cml_cc3d.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/run_cml_cc3d.bat @ONLY)

  configure_file(${RUN_SCRIPT_DIR}/compucell3d.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.bat @ONLY)  
  configure_file(${RUN_SCRIPT_DIR}/runScript.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.bat @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/paramScan.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.bat @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/twedit++.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.bat @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/optimization.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat @ONLY)

  configure_file(${RUN_SCRIPT_DIR}/compucell3d.bat.in.v2 ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.bat.in.v2 @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/twedit++.bat.in.v2 ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.bat.in.v2 @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/paramScan.bat.in.v2 ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.bat.in.v2 @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/runScript.bat.in.v2 ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.bat.in.v2 @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/optimization.bat.in.v2 ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat.in.v2 @ONLY)

  INSTALL(FILES
    ${CMAKE_BINARY_DIR}/post_install_tasks/cc3d.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/run_cml_cc3d.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat

    DESTINATION 

    ${CMAKE_INSTALL_PREFIX}
)


  if(BUILD_CPP_ONLY_EXECUTABLE)
    configure_file(${RUN_SCRIPT_DIR}/cc3dScript.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.bat @ONLY)
    INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.bat DESTINATION ${CMAKE_INSTALL_PREFIX})

  endif(BUILD_CPP_ONLY_EXECUTABLE)

  #copying SetupScript file
  INSTALL(FILES "${CMAKE_SOURCE_DIR}/core/post_install_tasks/windows/scriptSetup.py" DESTINATION ${CMAKE_INSTALL_PREFIX})



  INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.bat.in.v2 DESTINATION ${CMAKE_INSTALL_PREFIX})
  INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.bat.in.v2 DESTINATION ${CMAKE_INSTALL_PREFIX})
  INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.bat.in.v2 DESTINATION ${CMAKE_INSTALL_PREFIX})
  INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.bat.in.v2 DESTINATION ${CMAKE_INSTALL_PREFIX})
  INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat.in.v2 DESTINATION ${CMAKE_INSTALL_PREFIX})

elseif (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

    if (BUNDLE_PYTHON)
        SET(PYTHON_EXEC_FILE "\${PREFIX_CC3D}/python37/bin/python")
        message ("USING BUNDLE")
    else(BUNDLE_PYTHON)
        SET(PYTHON_EXEC_FILE ${PYTHON_EXECUTABLE})
        message ("USING EXTERNAL PYTHON")
    endif(BUNDLE_PYTHON)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/compucell3d.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/compucell3d.command @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/twedit++.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/twedit++.command @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/runScript.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/runScript.command @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/paramScan.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/paramScan.command @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/optimization.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/optimization.command @ONLY)


    FILE(GLOB mac_command_files "${CMAKE_BINARY_DIR}/post_install_tasks/mac/*.command")
    INSTALL(FILES ${mac_command_files} DESTINATION ${CMAKE_INSTALL_PREFIX}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE  WORLD_READ WORLD_WRITE WORLD_EXECUTE
    )

else()

  set(RUN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/linux)
    if (BUNDLE_PYTHON)
        SET(PYTHON_EXEC_FILE "\${PREFIX_CC3D}/python37/bin/python")
        message("USING BUNDLE")
    else (BUNDLE_PYTHON)
        SET(PYTHON_EXEC_FILE ${PYTHON_EXECUTABLE})
        message("USING EXTERNAL PYTHON")
    endif(BUNDLE_PYTHON)

    configure_file(${RUN_SCRIPT_DIR}/compucell3d.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.sh @ONLY)
    configure_file(${RUN_SCRIPT_DIR}/runScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.sh @ONLY)
    configure_file(${RUN_SCRIPT_DIR}/paramScan.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.sh @ONLY)
    configure_file(${RUN_SCRIPT_DIR}/twedit++.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.sh @ONLY)


    INSTALL(FILES
        ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.sh
        ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.sh
        ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.sh
        ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.sh
        DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                        GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                        WORLD_READ WORLD_WRITE WORLD_EXECUTE
        )

    if(BUILD_CPP_ONLY_EXECUTABLE)
      configure_file(${RUN_SCRIPT_DIR}/cc3dScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh @ONLY)
        INSTALL(FILES
            ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh
            DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                            WORLD_READ WORLD_WRITE WORLD_EXECUTE
            )

    endif(BUILD_CPP_ONLY_EXECUTABLE)


endif()

INSTALL(DIRECTORY  "${CMAKE_SOURCE_DIR}/core/Demos" DESTINATION ${CMAKE_INSTALL_PREFIX}
    PATTERN "*.svn" EXCLUDE
    PATTERN "${CMAKE_SOURCE_DIR}/core/Demos/*")


# # elseif (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

# #         message ("CMAKE_SYSTEM_VERSION " ${CMAKE_SYSTEM_VERSION})

# #         EXEC_PROGRAM(sw_vers ARGS -productVersion  OUTPUT_VARIABLE MAC_OS_PRODUCT_VERSION)
# #         STRING(REGEX REPLACE
# #             "([0-9]+).([0-9]+).([0-9]+)"
# #             "\\1;\\2;\\3" RESULT ${MAC_OS_PRODUCT_VERSION})
# #         message("RESULT =" ${RESULT})

# #         LIST(LENGTH RESULT n)
# #         MESSAGE("n=${n}")
# #         LIST(GET RESULT 0 MAJOR_OS_VERSION)
# #         # LIST(GET RESULT 1 MINOR_OS_VERSION)
# #         #LIST(GET RESULT 2 MINOR_MINOR_OS_VERSION)
# #         MESSAGE ("MAJOR_OS_VERSION=" ${MAJOR_OS_VERSION})
# #         # MESSAGE ("MINOR_OS_VERSION=" ${MINOR_OS_VERSION})
# #         #MESSAGE ("MINOR_MINOR_OS_VERSION=" ${MINOR_MINOR_OS_VERSION})

# #         # SET(MAC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mac)
# #         # EXEC_PROGRAM(sw_vers ARGS -productVersion  OUTPUT_VARIABLE DARWIN_VERSION)

# #         # EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
# #         # STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
# #         # MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")
# #         SET (PYTHON_MINOR_VERSION 7 CACHE STRING "Minor version of Python on OSX")

# #         # IF (${MINOR_OS_VERSION} GREATER 5)
# #         #   SET(PYTHON_MINOR_VERSION 6)
# #         # ELSE()
# #         #   SET(PYTHON_MINOR_VERSION 5)
# #         # ENDIF ()

# #         message ("PYTHON_MINOR_VERSION = "  ${PYTHON_MINOR_VERSION})



# #         if(${PYQT_VERSION} STREQUAL 4)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/compucell3d.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/compucell3d.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/twedit++.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/twedit++.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/runScript.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/runScript.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/paramScan.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/paramScan.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/celldraw.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/celldraw.command @ONLY)

# #         elseif(${PYQT_VERSION} STREQUAL 5)

# #             if (BUNDLE_PYTHON)
# #                 SET(PYTHON_EXEC_FILE "\${PREFIX_CC3D}/python27/bin/python2.7")
# #                 message ("USING BUNDLE")
# #             else(BUNDLE_PYTHON)
# #                 SET(PYTHON_EXEC_FILE ${PYTHON_EXECUTABLE})
# #                 message ("USING EXTERNAL PYTHON")
# #             endif(BUNDLE_PYTHON)

# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/pyqt5/compucell3d.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/compucell3d.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/pyqt5/twedit++.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/twedit++.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/pyqt5/runScript.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/runScript.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/pyqt5/paramScan.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/paramScan.command @ONLY)
# #             configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mac/pyqt5/optimization.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/optimization.command @ONLY)

# #         endif()


# #         FILE(GLOB mac_command_files "${CMAKE_BINARY_DIR}/post_install_tasks/mac/*.command")
# #         INSTALL(FILES ${mac_command_files} DESTINATION ${CMAKE_INSTALL_PREFIX}
# #           PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE  WORLD_READ WORLD_WRITE WORLD_EXECUTE
# #           )

# #         # FILE(GLOB mac_command_files "${MAC_PATH}/*.command")
# #         # # message("mac_command_files=" ${mac_command_files})
# #         # INSTALL(FILES ${mac_command_files} DESTINATION ${CMAKE_INSTALL_PREFIX}
# #         #   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE  WORLD_READ WORLD_WRITE WORLD_EXECUTE
# #         #   )


# #     if(BUILD_CPP_ONLY_EXECUTABLE)
# #       configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cc3dScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh @ONLY)
# #         INSTALL(FILES
# #             ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh
# #             DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
# #         )

# #     endif(BUILD_CPP_ONLY_EXECUTABLE)

# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# 	set(RUN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/linux)
#     if (BUNDLE_PYTHON)
#         SET(PYTHON_EXEC "\${PREFIX_CC3D}/Python27/bin/python")
#         message ("USING BUNDLE")
#         if (EXISTS ${DEPENDENCIES_DIR})

#             # note for boolean options we do not use ${}
#             if (DEPENDENCIES_COPY_FLAG)
#                 message("WILL COPY DEPENDENCIES")
#                 INSTALL(
#                     DIRECTORY  "@DEPENDENCIES_DIR@/"
#                     DESTINATION ${CMAKE_INSTALL_PREFIX}
#                     USE_SOURCE_PERMISSIONS
#                     PATTERN "@DEPENDENCIES_DIR@/*"
#                 )

#             endif()

#             message("DEPENDENCIES DIR ${DEPENDENCIES_DIR}")

#         else()
#             message( SEND_ERROR "\nIf BUNDLE_PYTHON option is checked you must specify DEPENDENCIES_DIR.\nYou also must make sure that DEPENDENCIES_DIR contains Python27 subfolder with full python distribution. You may use symbolic links ( call it Python27) to the full python distribution.
# Keep In mind that you also need to check DEPENDENCIES_COPY flag to enable copying of the dependencies into installation directory
#             " )
#         endif()


#     else(BUNDLE_PYTHON)
#         message ("USING SYSTEM")
#         #SET(PYTHON_EXEC "python")
#         SET(PYTHON_EXEC ${PYTHON_EXECUTABLE}) # provided by build-in cmake macro that locates python
#     endif()


#     # # configure_file(${RUN_SCRIPT_DIR}/compucell3d_old.sh.in.linux ${CMAKE_BINARY_DIR}/compucell3d_old.sh @ONLY)
#     # configure_file(${RUN_SCRIPT_DIR}/compucell3d.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.sh @ONLY)
#     # configure_file(${RUN_SCRIPT_DIR}/runScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.sh @ONLY)
#     # configure_file(${RUN_SCRIPT_DIR}/paramScan.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.sh @ONLY)
#     # configure_file(${RUN_SCRIPT_DIR}/twedit++.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.sh @ONLY)
#     # # configure_file(${RUN_SCRIPT_DIR}/celldraw.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/celldraw.sh @ONLY)
#     # configure_file(${RUN_SCRIPT_DIR}/optimization.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.sh @ONLY)


#     #     INSTALL(FILES
#     #         ${CMAKE_BINARY_DIR}/post_install_tasks/compucell3d.sh
#     #         ${CMAKE_BINARY_DIR}/post_install_tasks/runScript.sh
#     #         ${CMAKE_BINARY_DIR}/post_install_tasks/paramScan.sh
#     #         ${CMAKE_BINARY_DIR}/post_install_tasks/twedit++.sh
#     #         ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.sh
# 	# 		# ${CMAKE_BINARY_DIR}/post_install_tasks/celldraw.sh
#     #         DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#     #                                                         GROUP_READ GROUP_WRITE GROUP_EXECUTE
#     #                                                         WORLD_READ WORLD_WRITE WORLD_EXECUTE
#     #         )

#     if(BUILD_CPP_ONLY_EXECUTABLE)
#       configure_file(${RUN_SCRIPT_DIR}/cc3dScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh @ONLY)
#         INSTALL(FILES
#             ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh
#             DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#                                                             GROUP_READ GROUP_WRITE GROUP_EXECUTE
#                                                             WORLD_READ WORLD_WRITE WORLD_EXECUTE
#             )

#     endif(BUILD_CPP_ONLY_EXECUTABLE)


# #     file(COPY ${CMAKE_INSTALL_PREFIX}/compucell3d.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/compucell3d.sh
# #          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
# #                          GROUP_READ GROUP_WRITE GROUP_EXECUTE
# #                          WORLD_READ WORLD_WRITE WORLD_EXECUTE
# #     )
# #     configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runScript.sh.in.linux ${CMAKE_INSTALL_PREFIX}/runScript.sh @ONLY)
# #     if(BUILD_CPP_ONLY_EXECUTABLE)
# #       configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cc3dScript.sh.in.linux ${CMAKE_INSTALL_PREFIX}/cc3dScript.sh @ONLY)
# #     endif(BUILD_CPP_ONLY_EXECUTABLE)


#     #file(INSTALL DESTINATION ${CMAKE_INSTALL_PREFIX} TYPE FILE FILES ${CMAKE_CURRENT_SOURCE_DIR}/compucell3d.sh.in.linux )

# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# # copying vtk, Qt, python25,OPENGL libraries used by the player on Windows
# # if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#     # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/bin/" DESTINATION @CMAKE_INSTALL_PREFIX@/bin
#         # PATTERN "*.svn" EXCLUDE
#         # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/bin/*")
# # endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# # copying vtk, Qt, python25,OPENGL libraries used by the player on Windows
# if(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)
#     if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
#         if(${PYQT_VERSION} STREQUAL 4)
#             INSTALL(DIRECTORY  ${WINDOWS_DEPENDENCIES}/bin/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#                 PATTERN "*.svn" EXCLUDE
#                 PATTERN "${WINDOWS_DEPENDENCIES}/bin/*")

#             file(GLOB prerequisites "${WINDOWS_DEPENDENCIES}/Prerequisites/*")
#             list(REMOVE_ITEM prerequisites "${WINDOWS_DEPENDENCIES}/Prerequisites/.svn")
#             INSTALL(FILES ${prerequisites} DESTINATION "${CMAKE_INSTALL_PREFIX}/Prerequisites" )
#         elseif(${PYQT_VERSION} STREQUAL 5)
#             INSTALL(DIRECTORY  "${WINDOWS_DEPENDENCIES}/" DESTINATION ${CMAKE_INSTALL_PREFIX}
#                 PATTERN "*.svn" EXCLUDE
#                 PATTERN "${WINDOWS_DEPENDENCIES}/*")

#         endif()
#     endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

# endif(WINDOWS_DEPENDENCIES_INSTALL_ENABLE)


# # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Configuration/Version.py.in @CMAKE_SOURCE_DIR@/core/pythonSetupScripts/Version.py)


# # ADD_CUSTOM_COMMAND(
#     # OUTPUT Version.py
#     # COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/GetSVNRevisionGenerator.py
#     # # DEPENDS _CompuCell
#     # WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/
# # )

# # ADD_CUSTOM_TARGET(RevisionSVN ALL DEPENDS GetSVNRevision.py)

# SET(COMPUCELL3D_REVISION 0)
# EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${TOP_LEVEL_DIR}/cc3d ARGS RevisionNumber.py
#       OUTPUT_VARIABLE COMPUCELL3D_REVISION)


# # if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# # EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts ARGS ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/SVNRevisionWindows.py  ${CMAKE_SOURCE_DIR}
#       # OUTPUT_VARIABLE COMPUCELL3D_REVISION)
# # else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
# # EXEC_PROGRAM(${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts ARGS ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/SVNRevision.py
#       # OUTPUT_VARIABLE COMPUCELL3D_REVISION)

# # endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)


# # MESSAGE("COMPUCELL3D_REVISION= ",${COMPUCELL3D_REVISION})

# # configure_file(${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/Version.py.in @CMAKE_INSTALL_PREFIX@/pythonSetupScripts/Version.py)
# # configure_file(${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/Version.py.in ${CMAKE_SOURCE_DIR}/core/pythonSetupScripts/Version.py)
# # INSTALL(FILES  "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts/Version.py" DESTINATION @CMAKE_INSTALL_PREFIX@/pythonSetupScripts)

# configure_file(${TOP_LEVEL_DIR}/cc3d/Version.py.in ${TOP_LEVEL_DIR}/cc3d/Version.py)
# INSTALL(FILES  ${TOP_LEVEL_DIR}/cc3d/Version.py DESTINATION ${SITE_PACKAGES_INSTALL}/cc3d)


# # file(GLOB GetSVNRevision "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts/GetSVNRevision.py")
# # INSTALL(FILES ${GetSVNRevision} DESTINATION "@CMAKE_INSTALL_PREFIX@/pythonSetupScripts")

# # INSTALL(FILES ${GetSVNRevision} DESTINATION "@CMAKE_INSTALL_PREFIX@/player/Configuration")
# # INSTALL(FILES ${GetSVNRevision} DESTINATION "@CMAKE_INSTALL_PREFIX@/Twedit++")


# # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts" DESTINATION @CMAKE_INSTALL_PREFIX@
# #     PATTERN "*.svn" EXCLUDE
# #     PATTERN "CompuCellPythonSimulationCML.py" EXCLUDE
# #     PATTERN "Version.py.in*" EXCLUDE
# #     PATTERN "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts/*")



# # if (INSTALL_PLAYER)

# #     INSTALL(FILES  "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts/CompuCellPythonSimulationCML.py" DESTINATION @CMAKE_INSTALL_PREFIX@/${player_dir} )
# #     INSTALL(FILES  "@CMAKE_SOURCE_DIR@/core/pythonSetupScripts/ParameterScanRun.py" DESTINATION @CMAKE_INSTALL_PREFIX@/${player_dir} )

# # endif (INSTALL_PLAYER)


# INSTALL(DIRECTORY  "${CMAKE_SOURCE_DIR}/core/Demos" DESTINATION ${CMAKE_INSTALL_PREFIX}
#     PATTERN "*.svn" EXCLUDE
#     PATTERN "${CMAKE_SOURCE_DIR}/core/Demos/*")

# # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/core/examples_PythonTutorial" DESTINATION @CMAKE_INSTALL_PREFIX@
#     # PATTERN "*.svn" EXCLUDE
#     # PATTERN "@CMAKE_SOURCE_DIR@/core/examples_PythonTutorial/*")

# # Copying images
# INSTALL(DIRECTORY  "${CMAKE_SOURCE_DIR}/core/icons" DESTINATION ${CMAKE_INSTALL_PREFIX}
#     PATTERN "*.svn" EXCLUDE
#     PATTERN "${CMAKE_SOURCE_DIR}/core/icons/*")
