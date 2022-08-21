get_filename_component(_cc3d_install_prefix "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
set(_cc3d_cmake_prefix_path_cached ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${_cc3d_install_prefix})

set(_cc3d_config_paths )
if(NOT cc3d_FIND_COMPONENTS)
    set(cc3d_FIND_COMPONENTS )
    file(GLOB _cc3d_config_paths 
         LIST_DIRECTORIES false 
         "cc3d-*-config.cmake"
    )
    foreach(conf_path ${_cc3d_config_paths})
        string(REGEX MATCH "cc3d-*-config.cmake" comp ${conf_path})
        list(APPEND cc3d_FIND_COMPONENTS ${comp})
    endforeach()
endif()

foreach(comp ${cc3d_FIND_COMPONENTS})
    set(conf_path ${_cc3d_install_prefix}/cc3d-${comp}-config.cmake)
    if(EXISTS ${conf_path})
        include(${conf_path})
        set(cc3d_${comp}_FOUND True)
    else()
        set(_cc3d_message_type WARNING)
        if(${cc3d_FIND_REQUIRED})
            set(_cc3d_message_type FATAL_ERROR)
        endif()
        message(${_cc3d_message_type} "cc3d component ${comp} not found.")
        set(cc3d_${comp}_FOUND False)
    endif()
endforeach()

set(cc3d_FOUND True)

set(CMAKE_PREFIX_PATH ${_cc3d_cmake_prefix_path_cached})