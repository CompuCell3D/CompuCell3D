cmake_minimum_required(VERSION 2.8)
project(rr_support)
set(SUPPORT_FOLDER 			${CMAKE_CURRENT_SOURCE_DIR})

#=== Install support code for on the fly compilation
FILE(GLOB rr_support_headers *.h)
FILE(GLOB rr_support_source *.c)
install(FILES 
${rr_support_headers} 
${rr_support_source} 
        DESTINATION rr_support
        COMPONENT rr_core
) 

