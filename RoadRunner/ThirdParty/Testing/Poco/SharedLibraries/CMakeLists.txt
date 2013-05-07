cmake_minimum_required(VERSION 2.8)
set(target poco_shared_lib_test)

add_executable(${target} 
main.cpp 
)

add_definitions(
-DPOCO_STATIC
-DPOCO_NO_AUTOMATIC_LIBS
#-DWIN32
)

if(BUILD_WITH_SHARED_LIBS) 
target_link_libraries(${target} 
poco_foundation
)
else()
target_link_libraries (${target} 
poco_foundation-static
)
if(UNIX)
target_link_libraries (${target} 
dl
pthread
)
endif()
endif()

add_subdirectory(TestDLL)


install (TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT apps)
