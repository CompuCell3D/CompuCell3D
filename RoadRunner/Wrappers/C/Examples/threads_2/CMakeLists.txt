project(threads_2 C)
set(target threads_2)

add_executable( ${target} 
main.c
)

target_link_libraries (${target} 
rrc_api)

if(LINUX)
target_link_libraries (${target} 
blas
lapack
f2c
pthread
)


endif()

install(
TARGETS ${target} 		
DESTINATION bin 
COMPONENT compiled_examples
)
