CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(target SharedLibTest) 

set(src 
TestLibrary
)

add_library(${target} SHARED ${src})

set_property(TARGET ${target}  
 PROPERTY  COMPILE_DEFINITIONS 
 EXPORT_POCO
) 

install(    
            TARGETS ${target} 
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            COMPONENT Testing
)            


