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
set(cc3d_py_source_dir ${CMAKE_SOURCE_DIR}/../cc3d)

message("THIS IS cc3d_py_source_dir: " ${cc3d_py_source_dir})

# Setting variables for configuring files
if(BUILD_STANDALONE)
    if (${CMAKE_SYSTEM_NAME} STREQUAL Windows)
        SET(PYTHON_EXEC_FILE "${CMAKE_INSTALL_PREFIX}/python37/python")
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
        SET(PYTHON_ENV_PREFIX "${CMAKE_INSTALL_PREFIX}/python37")
        SET(APP_INSTALL_PREFIX ${PYTHON_ENV_PREFIX})
        SET(PYTHON_EXEC_FILE "${CMAKE_INSTALL_PREFIX}/python37/bin/python")
    else()
        SET(PYTHON_EXEC_FILE "${CMAKE_INSTALL_PREFIX}/python37/bin/python")
    endif()
    set(NAME_INSTALL_PREFIX )
    message("USING BUNDLE")
else(BUILD_STANDALONE)
    if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var(name='prefix'))"
            OUTPUT_VARIABLE PYTHON_ENV_PREFIX
        )
        SET(APP_INSTALL_PREFIX ${PYTHON_ENV_PREFIX}/bin)
    endif()
    SET(PYTHON_EXEC_FILE python)
    set(NAME_INSTALL_PREFIX cc3d_)
    message("USING EXTERNAL PYTHON")
endif(BUILD_STANDALONE)

if(BUILD_PYINTERFACE)
    INSTALL(FILES  ${cc3d_py_source_dir}/__init__.py DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR})
    INSTALL(FILES  ${cc3d_py_source_dir}/run_script.py DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR})

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/CompuCellSetup DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/CompuCellSetup/*
    )

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/core DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/core/*
    )

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/doc DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/doc/*)

    INSTALL(DIRECTORY  ${cc3d_py_source_dir}/tests DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR}
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "*.bak" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN ${cc3d_py_source_dir}/tests/*)
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

  # configure_file(${RUN_SCRIPT_DIR}/run_cml_cc3d.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/run_cml_cc3d.bat @ONLY)

  configure_file(${RUN_SCRIPT_DIR}/runScript.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}runScript.bat @ONLY)
  configure_file(${RUN_SCRIPT_DIR}/paramScan.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}paramScan.bat @ONLY)
  # configure_file(${RUN_SCRIPT_DIR}/optimization.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat @ONLY)

  INSTALL(FILES
    # ${CMAKE_BINARY_DIR}/post_install_tasks/run_cml_cc3d.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}runScript.bat
    ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}paramScan.bat
    # ${CMAKE_BINARY_DIR}/post_install_tasks/optimization.bat

    DESTINATION
    ${COMPUCELL3D_INSTALL_SCRIPT_DIR}
)


  if(BUILD_CPP_ONLY_EXECUTABLE)
    configure_file(${RUN_SCRIPT_DIR}/cc3dScript.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.bat @ONLY)
    INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.bat DESTINATION ${COMPUCELL3D_INSTALL_SCRIPT_DIR})

  endif(BUILD_CPP_ONLY_EXECUTABLE)

elseif (${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

    set(RUN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mac)

    configure_file(${RUN_SCRIPT_DIR}/runScript.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/${NAME_INSTALL_PREFIX}runScript.command @ONLY)
    configure_file(${RUN_SCRIPT_DIR}/paramScan.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/${NAME_INSTALL_PREFIX}paramScan.command @ONLY)
    # configure_file(${RUN_SCRIPT_DIR}/optimization.command.in ${CMAKE_BINARY_DIR}/post_install_tasks/mac/optimization.command @ONLY)


    # FILE(GLOB mac_command_files "${CMAKE_BINARY_DIR}/post_install_tasks/mac/*.command")
    INSTALL(FILES
        ${CMAKE_BINARY_DIR}/post_install_tasks/mac/${NAME_INSTALL_PREFIX}runScript.command
        ${CMAKE_BINARY_DIR}/post_install_tasks/mac/${NAME_INSTALL_PREFIX}paramScan.command
        # ${CMAKE_BINARY_DIR}/post_install_tasks/mac/optimization.command
        DESTINATION ${COMPUCELL3D_INSTALL_SCRIPT_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE  WORLD_READ WORLD_WRITE WORLD_EXECUTE
    )

else()

  set(RUN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/linux)

    configure_file(${RUN_SCRIPT_DIR}/runScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}runScript.sh @ONLY)
    configure_file(${RUN_SCRIPT_DIR}/paramScan.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}paramScan.sh @ONLY)

    INSTALL(FILES
        ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}runScript.sh
        ${CMAKE_BINARY_DIR}/post_install_tasks/${NAME_INSTALL_PREFIX}paramScan.sh
        DESTINATION ${COMPUCELL3D_INSTALL_SCRIPT_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                                  GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                                  WORLD_READ WORLD_WRITE WORLD_EXECUTE
        )

    if(BUILD_CPP_ONLY_EXECUTABLE)
      configure_file(${RUN_SCRIPT_DIR}/cc3dScript.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh @ONLY)
        INSTALL(FILES
            ${CMAKE_BINARY_DIR}/post_install_tasks/cc3dScript.sh
            DESTINATION ${COMPUCELL3D_INSTALL_SCRIPT_DIR}
            PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_WRITE GROUP_EXECUTE
            WORLD_READ WORLD_WRITE WORLD_EXECUTE
        )

    endif(BUILD_CPP_ONLY_EXECUTABLE)

endif()

set(cc3d_install_dir_full ${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_PY_MODULE_DIR})
file(RELATIVE_PATH cc3d_install_prefix_rel ${cc3d_install_dir_full} ${CMAKE_INSTALL_PREFIX})
file(RELATIVE_PATH cc3d_scripts_path_rel ${cc3d_install_dir_full} ${COMPUCELL3D_INSTALL_SCRIPT_DIR_FULL_PATH})
file(RELATIVE_PATH cc3d_cpp_path_rel ${cc3d_install_dir_full} ${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR})
if(BUILD_PYINTERFACE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.py.in ${CMAKE_BINARY_DIR}/post_install_tasks/config.py @ONLY)
    INSTALL(FILES ${CMAKE_BINARY_DIR}/post_install_tasks/config.py DESTINATION ${COMPUCELL3D_INSTALL_PY_MODULE_DIR})
endif()

INSTALL(FILES ${CMAKE_SOURCE_DIR}/cmake/cc3d-config.cmake DESTINATION ${COMPUCELL3D_INSTALL_CONFIG_DIR})

if(BUILD_STANDALONE)
    INSTALL(DIRECTORY  "${CMAKE_SOURCE_DIR}/core/Demos" DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN "*.svn" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN "${CMAKE_SOURCE_DIR}/core/Demos/*"
    )
endif()
