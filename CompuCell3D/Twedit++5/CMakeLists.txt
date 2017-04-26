# CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
# PROJECT(Twedit++)


SET (INSTALLATION_DIRECTORY @CMAKE_INSTALL_PREFIX@/Twedit++)
#Copying Twedit++ files
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/ActionManager" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/ActionManager/*")

INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/APIs" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/APIs/*")
    
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/CQt" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/CQt/*")
    
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/Encoding" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/Encoding/*")

INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/icons" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/icons/*")
    
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/Messaging" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/Messaging/*")
    
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/PluginManager" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/PluginManager/*")

INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/Plugins" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "*.information" EXCLUDE
    PATTERN "*.qrc" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/Plugins/*")
    
INSTALL(DIRECTORY  "@CMAKE_CURRENT_SOURCE_DIR@/themes" DESTINATION ${INSTALLATION_DIRECTORY}
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_CURRENT_SOURCE_DIR@/themes/*")    
    
    
FILE(GLOB pif_py_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
FILE(GLOB pif_pyw_files "${CMAKE_CURRENT_SOURCE_DIR}/*.pyw")
# LIST(APPEND pif_python_files ${Mac_CPP} ${Mac_H})

# MESSAGE("PIF_PYTHON_FILES = " ${pif_py_files} )
INSTALL(FILES   ${pif_pyw_files} ${pif_py_files} DESTINATION ${INSTALLATION_DIRECTORY})

# #setting up run scripts

# if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  # #here I need to change "/" to "\" to comply with windows style paths
  # set(PIFGEN_INSTALL_DIR_WINDOWS_STYLE ${CMAKE_INSTALL_PREFIX})
  # #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  # #for more info please see http://www.amk.ca/python/howto/regex/ 
  # STRING(REGEX REPLACE "/" "\\\\" PIFGEN_INSTALL_DIR_WINDOWS_STYLE ${PIFGEN_INSTALL_DIR_WINDOWS_STYLE})

  
  # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/post_install_tasks/pifgenerator.bat.in.windows ${CMAKE_INSTALL_PREFIX}/pifgenerator.bat @ONLY)
  
  
  # #copying SetupScript file
  # file(GLOB pifgenerator.bat_in_v2 "@CMAKE_CURRENT_SOURCE_DIR@/post_install_tasks/pifgenerator.bat.in.v2")
  # file(INSTALL DESTINATION "@CMAKE_INSTALL_PREFIX@" TYPE FILE FILES ${pifgenerator.bat_in_v2})

  
  
# else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pifgenerator.sh.in.linux ${CMAKE_BINARY_DIR}/pifgenerator.sh @ONLY)


        # INSTALL(FILES  
            # ${CMAKE_BINARY_DIR}/pifgenerator.sh 
            # DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                            # GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                            # WORLD_READ WORLD_WRITE WORLD_EXECUTE
            # )

    
# endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    


# # Local Variables:
# # coding: US-ASCII
# # End:
