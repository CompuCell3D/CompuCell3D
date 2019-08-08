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


