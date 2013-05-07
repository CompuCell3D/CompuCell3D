cmake_minimum_required(VERSION 2.6)
project(poco)

#if(${MSVC})
if(WIN32)
add_definitions(
#-DUSRDLL
-DWIN32
-D_WIN32
-DPCRE_STATIC
-DPOCO_STATIC
)
endif()

add_subdirectory(Foundation)
add_subdirectory(XML)
