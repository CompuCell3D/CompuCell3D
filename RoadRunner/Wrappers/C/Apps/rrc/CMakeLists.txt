cmake_minimum_required(VERSION 2.8)
set(target rrc)

add_executable(${target} 
main.cpp 
Args.cpp
GetOptions.cpp
)

add_definitions(
-DPOCO_STATIC
-DSTATIC_PUGI
-DLIBSBML_STATIC
-DSTATIC_RR
-DSTATIC_LIBSTRUCT
-DLIBLAX_STATIC
-DSTATIC_NLEQ
)

target_link_libraries (${target} 
rrc_api
)

#Not sure why linux linker need all libraries at linktime?? All symbols 'should' be in roadrunner-static
if(UNIX)
target_link_libraries (${target} 
f2c
lapack
blas
dl
z
xml2
pthread
)
endif()

install(    
TARGETS ${target} 		
DESTINATION bin COMPONENT apps
)
