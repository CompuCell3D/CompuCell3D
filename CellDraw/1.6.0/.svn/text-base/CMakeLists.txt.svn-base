CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(CellDraw)


SET (INSTALLATION_DIRECTORY @CMAKE_INSTALL_PREFIX@)
# #Copying PIF_Generator files
# INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/generated_test_PIFs" DESTINATION ${INSTALLATION_DIRECTORY}
    # PATTERN "*.svn" EXCLUDE
    # PATTERN "*.ui" EXCLUDE
    # PATTERN "*.in" EXCLUDE
    # PATTERN "@CMAKE_SOURCE_DIR@/generated_test_PIFs/*")

INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/icons" DESTINATION ${INSTALLATION_DIRECTORY}/CellDraw
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_SOURCE_DIR@/icons/*")

INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/src/cc3Dhelpfiles" DESTINATION ${INSTALLATION_DIRECTORY}/CellDraw
    PATTERN "*.svn" EXCLUDE
    PATTERN "*.ui" EXCLUDE
    PATTERN "*.in" EXCLUDE
    PATTERN "@CMAKE_SOURCE_DIR@/icons/*")

    
# INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/input_images_examples" DESTINATION ${INSTALLATION_DIRECTORY}
    # PATTERN "*.svn" EXCLUDE
    # PATTERN "*.ui" EXCLUDE
    # PATTERN "*.in" EXCLUDE
    # PATTERN "@CMAKE_SOURCE_DIR@/input_images_examples/*")

# INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/PIF_Generator_manual" DESTINATION ${INSTALLATION_DIRECTORY}
    # PATTERN "*.svn" EXCLUDE
    # PATTERN "*.ui" EXCLUDE
    # PATTERN "*.in" EXCLUDE
    # PATTERN "@CMAKE_SOURCE_DIR@/PIF_Generator_manual/*")
    
FILE(GLOB pif_py_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.py")
FILE(GLOB pif_pyw_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.pyw")
# LIST(APPEND pif_python_files ${Mac_CPP} ${Mac_H})

# MESSAGE("PIF_PYTHON_FILES = " ${pif_py_files} )
INSTALL(FILES   ${pif_pyw_files} ${pif_py_files} DESTINATION ${INSTALLATION_DIRECTORY}/CellDraw)

#setting up run scripts

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
  #here I need to change "/" to "\" to comply with windows style paths
  set(CELLDRAW_INSTALL_DIR_WINDOWS_STYLE ${CMAKE_INSTALL_PREFIX})
  #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
  #for more info please see http://www.amk.ca/python/howto/regex/ 
  STRING(REGEX REPLACE "/" "\\\\" CELLDRAW_INSTALL_DIR_WINDOWS_STYLE ${CELLDRAW_INSTALL_DIR_WINDOWS_STYLE})

  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/post_install_tasks/celldraw.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/celldraw.bat @ONLY)
  
  
  #copying SetupScript file
  file(GLOB celldraw.bat_in_v2 "@CMAKE_SOURCE_DIR@/post_install_tasks/celldraw.bat.in.v2")

  
 INSTALL(FILES ${celldraw.bat_in_v2} ${CMAKE_BINARY_DIR}/post_install_tasks/celldraw.bat  DESTINATION ${INSTALLATION_DIRECTORY})

  

  
  
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/celldraw.sh.in.linux ${CMAKE_BINARY_DIR}/celldraw.sh @ONLY)


        INSTALL(FILES  
            ${CMAKE_BINARY_DIR}/celldraw.sh 
            DESTINATION ${CMAKE_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                            WORLD_READ WORLD_WRITE WORLD_EXECUTE
            )

    
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    


# Local Variables:
# coding: US-ASCII
# End:
