#SET_DIRECTORY_PROPERTIES(PROPERTIES INCLUDE_DIRECTORIES ${PYQTPLAYER_SOURCE_DIR})
#SET(CC3D_DIR "/home/dexity/CompuCellBin")
#SET(CC3D_LIB "${CC3D_DIR}/lib")
SET(PYQTPLAYER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(VTK REQUIRED)

INCLUDE_DIRECTORIES ( 
    ${VTK_INCLUDE_DIRS}
    ${COMPUCELL3D_SOURCE_DIR}/core
    ${COMPUCELL3D_SOURCE_DIR}/core/CompuCell3D
    )


#Copying player files


if(INSTALL_PLAYER)

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Configuration" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.ui" EXCLUDE
        PATTERN "*.in" EXCLUDE
        PATTERN "*.pyc" EXCLUDE        
        PATTERN "@CMAKE_SOURCE_DIR@/player/Configuration/*")
        
    # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Configuration/Version.py.in @CMAKE_INSTALL_PREFIX@/player/Configuration/Version.py)    




    # configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Configuration/Version.py.in @CMAKE_BINARY_DIR@/player/Configuration/Version.py)   
    # INSTALL(FILES @CMAKE_BINARY_DIR@/player/Configuration/Version.py DESTINATION @CMAKE_INSTALL_PREFIX@/player/Configuration/)     


    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/CQt" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/CQt/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/CustomGui" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/CustomGui/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/icons" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/icons/*")
        
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Messaging" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/Messaging/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/PluginManager" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/PluginManager/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Plugins" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/Plugins/*")
        
    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Simulation" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/Simulation/*")    

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/UI" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/UI/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Utilities" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/Utilities/*")

        
        
        
        
        
    # # #Copying Twedit++ plugin files that connect player with Twedit++
    # # file(GLOB pluginsScripts "@CMAKE_SOURCE_DIR@/player/PluginsTwedit++/*.py")
    # # list(REMOVE_ITEM pluginsScripts "@CMAKE_SOURCE_DIR@/player/PluginsTwedit++/.svn")
    # # INSTALL(FILES ${pluginsScripts} DESTINATION "@CMAKE_INSTALL_PREFIX@/Twedit++/Plugins" )     

    # # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/PluginsTwedit++/CompuCell3D" DESTINATION @CMAKE_INSTALL_PREFIX@/Twedit++/Plugins
        # # PATTERN "*.svn" EXCLUDE
        # # PATTERN "*.ui" EXCLUDE
        # # PATTERN "*.in" EXCLUDE
        # # PATTERN "@CMAKE_SOURCE_DIR@/player/PluginsTwedit++/CompuCell3D*")
        
        
        
    if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)   
        INSTALL(FILES  
            ${CMAKE_SOURCE_DIR}/player/Utilities/QVTKRenderWindowInteractor_mac.py 
            DESTINATION ${CMAKE_INSTALL_PREFIX}/player/Utilities RENAME QVTKRenderWindowInteractor.py
            )
        INSTALL(FILES  
            ${CMAKE_SOURCE_DIR}/player/Configuration_settings/_settings_osx.xml 
            DESTINATION ${CMAKE_INSTALL_PREFIX}/player/Configuration RENAME _settings.xml
            )
            
    else(${CMAKE_SYSTEM_NAME} STREQUAL Darwin)
        INSTALL(FILES  
            ${CMAKE_SOURCE_DIR}/player/Configuration_settings/_settings.xml 
            DESTINATION ${CMAKE_INSTALL_PREFIX}/player/Configuration
            )
        
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Darwin) 

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/ViewManager" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/ViewManager/*")

    INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/player/Graphics" DESTINATION @CMAKE_INSTALL_PREFIX@/player
        PATTERN "*.svn" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
        PATTERN "@CMAKE_SOURCE_DIR@/player/Graphics/*")
        
    # # copying PyQt, Sip and VTK libraries used by the player on Windows
    # if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)   
        # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/player/PyQt4" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/player/PyQt4/*")
            
        # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/player/sip" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/player/sip/*")
            
        # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/player/sip_utils" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/player/sip_utils/*")
            
        # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/player/vtk" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/player/vtk/*")
            
        # INSTALL(DIRECTORY  "@CMAKE_SOURCE_DIR@/windows_libs/player/VTKLibs" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "@CMAKE_SOURCE_DIR@/windows_libs/player/VTKLibs/*")
            
    # endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)    

    # copying PyQt, Sip and VTK libraries used by the player on Windows
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND WINDOWS_DEPENDENCIES_INSTALL_ENABLE)   
        INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/Python27" DESTINATION @CMAKE_INSTALL_PREFIX@
            PATTERN "*.svn" EXCLUDE
            PATTERN "*.pyc" EXCLUDE
            PATTERN "@WINDOWS_DEPENDENCIES@/Python27/*")
    
    # if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND WINDOWS_DEPENDENCIES_INSTALL_ENABLE)   
        # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/PyQt4" DESTINATION @CMAKE_INSTALL_PREFIX@/lib/PythonDeps
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "*.pyc" EXCLUDE
            # PATTERN "@WINDOWS_DEPENDENCIES@/player/PyQt4/*")

        # # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/numpy" DESTINATION @CMAKE_INSTALL_PREFIX@/player
            # # PATTERN "*.svn" EXCLUDE
            # # PATTERN "@WINDOWS_DEPENDENCIES@/player/numpy/*")
            
        # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/sip" DESTINATION @CMAKE_INSTALL_PREFIX@/lib/PythonDeps
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "*.pyc" EXCLUDE
            # PATTERN "@WINDOWS_DEPENDENCIES@/player/sip/*")
            
          # file(GLOB siputils "@WINDOWS_DEPENDENCIES@/player/sip_utils/*")
          # list(REMOVE_ITEM siputils "@WINDOWS_DEPENDENCIES@/player/sip_utils/.svn")
          # INSTALL(FILES ${siputils} DESTINATION "@CMAKE_INSTALL_PREFIX@/lib/PythonDeps" )  
            
            
        # # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/sip_utils" DESTINATION @CMAKE_INSTALL_PREFIX@/lib/PythonDeps
            # # PATTERN "*.svn" EXCLUDE
            # # PATTERN "@WINDOWS_DEPENDENCIES@/player/sip_utils/*")
            
        # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/vtk" DESTINATION @CMAKE_INSTALL_PREFIX@/lib/PythonDeps
            # PATTERN "*.svn" EXCLUDE
            # PATTERN "*.pyc" EXCLUDE
            # PATTERN "@WINDOWS_DEPENDENCIES@/player/vtk/*")
            
        # # INSTALL(DIRECTORY  "@WINDOWS_DEPENDENCIES@/player/VTKLibs" DESTINATION @CMAKE_INSTALL_PREFIX@/bin
            # # PATTERN "*.svn" EXCLUDE
            # # PATTERN "@WINDOWS_DEPENDENCIES@/player/VTKLibs/*")
            
    endif()    


    INSTALL(FILES  
        ${CMAKE_SOURCE_DIR}/player/compucell3d.pyw
        ${CMAKE_SOURCE_DIR}/player/DefaultData.py
        ${CMAKE_SOURCE_DIR}/player/plugins.txt
        DESTINATION ${CMAKE_INSTALL_PREFIX}/player
        )
endif(INSTALL_PLAYER)