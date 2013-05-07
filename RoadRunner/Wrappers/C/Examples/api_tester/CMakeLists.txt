set(target api_tester)

add_executable( ${target} 
main.cpp
)

target_link_libraries (${target} rrc_api)

install(
        TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT compiled_examples)
