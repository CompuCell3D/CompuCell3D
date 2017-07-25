
if(${PYQT_VERSION} STREQUAL 5)
    if(APPLE)
        SET(player_dir "player5")
        INSTALL(FILES @CMAKE_SOURCE_DIR@/${player_dir}/qt5_osxhelper_binary/libcpposxhelper.dylib
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${player_dir}/Utilities)
    endif(APPLE)
endif()
