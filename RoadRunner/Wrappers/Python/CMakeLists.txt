CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(rrWrappers)

if(BUILD_TESTS)
add_subdirectory(Testing)
endif()

if(INSTALL_EXAMPLES)
add_subdirectory(Examples)
endif()
#=============== INSTALL =====================
install(
        FILES 
        rrPython.py 
        DESTINATION python  	
        COMPONENT testing
       )

