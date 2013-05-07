#writing run scripts
set(LIBRARY_PATH_SYNTAX LD_LIBRARY_PATH) #default choice
if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
   set(LIBRARY_PATH_SYNTAX LD_LIBRARY_PATH)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Linux)

if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
   set(LIBRARY_PATH_SYNTAX DYLD_LIBRARY_PATH)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)

SET(COMPUCELL_INSTALL_DIR ${CC3D_INSTALL_PREFIX})

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    #here I need to change "/" to "\" to comply with windows style paths
    set(COMPUCELL_INSTALL_DIR_WINDOWS_STYLE ${COMPUCELL_INSTALL_DIR})
    #note to replace "/" with "\" you need in fact use "escaped baskslash as a string literal". this is property of regex
    #for more info please see http://www.amk.ca/python/howto/regex/ 
    STRING(REGEX REPLACE "/" "\\\\" COMPUCELL_INSTALL_DIR_WINDOWS_STYLE ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rrPython.bat.in.windows ${CMAKE_BINARY_DIR}/post_install_tasks/rrPython.bat @ONLY)



    INSTALL(FILES  ${CMAKE_BINARY_DIR}/post_install_tasks/rrPython.bat DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE})




    #copying SetupScript file
    INSTALL(FILES "@CMAKE_CURRENT_SOURCE_DIR@/rrPython.py" DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE})    
    #install demo SBML file
    INSTALL(FILES "@CMAKE_CURRENT_SOURCE_DIR@/test_1.xml" DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE})    

    #copying RoadRunnerSetup file  
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/RoadRunnerSetup.py DESTINATION ${CC3D_INSTALL_PREFIX})

    
    #installing tcc compil;er for windows
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/ThirdParty/compilers" DESTINATION @COMPUCELL_INSTALL_DIR@
          PATTERN "*.svn" EXCLUDE
          PATTERN "@CMAKE_SOURCE_DIR@/ThirdParty/compilers/*")

    #installing rr_support directory      
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/ThirdParty/rr_support" DESTINATION @COMPUCELL_INSTALL_DIR@
        PATTERN "*.svn" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/ThirdParty/rr_support/*")  
  
  
  
  #instaling roadrunner dll dependencies

   SET(THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE ${THIRD_PARTY_INSTALL_FOLDER})
   # STRING(REGEX REPLACE "/" "\\\\" THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE ${THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE})
  STRING(REGEX REPLACE "\\\\" "/" THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE ${THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE})
   
  
  INSTALL(FILES "@THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE@/bin/iconv.dll"  DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE}/bin)
  INSTALL(FILES "@THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE@/bin/libsbml.dll" DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE}/bin)
  INSTALL(FILES "@THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE@/bin/libxml2.dll" DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE}/bin)
  INSTALL(FILES "@THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE@/bin/zlib1.dll" DESTINATION  ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE}/bin)
  INSTALL(FILES "@THIRD_PARTY_INSTALL_FOLDER_WIN_STYLE@/bin/rr-libstruct.dll" DESTINATION ${COMPUCELL_INSTALL_DIR_WINDOWS_STYLE}/bin)
  
  # INSTALL(FILES "@CMAKE_SOURCE_DIR@/core/post_install_tasks/compucell3d.bat.in.v2" DESTINATION ${CMAKE_INSTALL_PREFIX})  
  # INSTALL(FILES "@CMAKE_SOURCE_DIR@/core/post_install_tasks/runScript.bat.in.v2" DESTINATION ${CMAKE_INSTALL_PREFIX})  
  # INSTALL(FILES "@CMAKE_SOURCE_DIR@/core/post_install_tasks/twedit++.bat.in.v2" DESTINATION ${CMAKE_INSTALL_PREFIX})  
  
 
  
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compucell3d_old.sh.in.linux ${CMAKE_BINARY_DIR}/compucell3d_old.sh @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rrPython.sh.in.linux ${CMAKE_BINARY_DIR}/post_install_tasks/rrPython.sh @ONLY)
    
        INSTALL(FILES  
            ${CMAKE_BINARY_DIR}/post_install_tasks/rrPython.sh 
            DESTINATION ${CC3D_INSTALL_PREFIX} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                                                            WORLD_READ WORLD_WRITE WORLD_EXECUTE
            )

            
    #installing rr_support directory      
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/ThirdParty/rr_support" DESTINATION @COMPUCELL_INSTALL_DIR@
        PATTERN "*.svn" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/ThirdParty/rr_support/*")  
            
      
      #copying rrPython file
      INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/rrPython.py DESTINATION ${CC3D_INSTALL_PREFIX})
          
      #install demo SBML file
      INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_1.xml DESTINATION ${CC3D_INSTALL_PREFIX})    

      #copying RoadRunnerSetup file  
      INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/RoadRunnerSetup.py DESTINATION ${CC3D_INSTALL_PREFIX})
            
      file(GLOB libsbmlfiles "${THIRD_PARTY_INSTALL_FOLDER}/lib/libsbml.*")
      file(GLOB lapackfiles "${THIRD_PARTY_INSTALL_FOLDER}/lib/liblapack.*")
      file(GLOB libstructfiles "${THIRD_PARTY_INSTALL_FOLDER}/lib/librr-libstruct.*")
      
      INSTALL(FILES ${libsbmlfiles} ${lapackfiles} ${libstructfiles} DESTINATION ${CC3D_INSTALL_PREFIX}/lib)    
      # INSTALL(DIRECTORY ${THIRD_PARTY_INSTALL_FOLDER}/lib DESTINATION ${CC3D_INSTALL_PREFIX} PATTERN ${THIRD_PARTY_INSTALL_FOLDER}/lib/libsbml*)    
              
      
      
      
      
#     file(COPY ${CMAKE_INSTALL_PREFIX}/compucell3d.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/compucell3d.sh 
#          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#                          GROUP_READ GROUP_WRITE GROUP_EXECUTE
#                          WORLD_READ WORLD_WRITE WORLD_EXECUTE
#     )
#     configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runScript.sh.in.linux ${CMAKE_INSTALL_PREFIX}/runScript.sh @ONLY)
#     if(BUILD_CPP_ONLY_EXECUTABLE)
#       configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cc3dScript.sh.in.linux ${CMAKE_INSTALL_PREFIX}/cc3dScript.sh @ONLY)
#     endif(BUILD_CPP_ONLY_EXECUTABLE)
    
    
    #file(INSTALL DESTINATION ${CMAKE_INSTALL_PREFIX} TYPE FILE FILES ${CMAKE_CURRENT_SOURCE_DIR}/compucell3d.sh.in.linux )
    
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

