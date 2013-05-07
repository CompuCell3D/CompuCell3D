cmake_minimum_required(VERSION 2.6)
project(pugixml)

set(target pugi)

set(HEADERS src/pugixml.hpp src/pugiconfig.hpp)
set(SOURCES ${HEADERS} src/pugixml.cpp)

#add_library(${target}  SHARED ${SOURCES})
#
#set_property(TARGET         ${target}  
#             PROPERTY    COMPILE_DEFINITIONS 
#                         SHARED_PUGI 
#            ) 

add_library(${target}-static  STATIC ${SOURCES})
set_property(TARGET         ${target}-static  
             PROPERTY    COMPILE_DEFINITIONS 
                         STATIC_PUGI 
            ) 

install(FILES ${HEADERS} 
        DESTINATION include/pugi)

#install(TARGETS ${target} 
#        LIBRARY DESTINATION lib 
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin)
#
install(TARGETS ${target}-static 
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib)

