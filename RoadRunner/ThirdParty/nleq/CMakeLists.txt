CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(nleq)

set(target nleq) 

set(headers nleq1.h f2c_nleq.h)

set(sources 
nleq1.c
wnorm.c
zibconst.c
zibmon.c
zibsec.c
)

add_library(${target}-static STATIC ${sources})

if(WIN32)
set_property(TARGET ${target}-static  
             PROPERTY  COMPILE_DEFINITIONS 
                       WIN32
                       STATIC_NLEQ
             ) 
else()
set_property(TARGET ${target}-static  
             PROPERTY  COMPILE_DEFINITIONS 
                       STATIC_NLEQ
             ) 

endif()

target_link_libraries (${target}-static f2c)


#Install required files
install(FILES ${headers} 
        DESTINATION include/nleq)

install(TARGETS ${target}-static 
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib
)

# add_library(${target} SHARED ${sources})

# if(WIN32)
# set_property(TARGET ${target}
             # PROPERTY  COMPILE_DEFINITIONS 
                       # WIN32
                       # STATIC_NLEQ
             # ) 
# else()
# set_property(TARGET ${target}
             # PROPERTY  COMPILE_DEFINITIONS 
                       # STATIC_NLEQ
             # ) 

# endif()

# target_link_libraries (${target} f2c)


# #Install required files
# install(FILES ${headers} 
        # DESTINATION include/nleq)

# install(TARGETS ${target} 
        # LIBRARY DESTINATION lib 
        # ARCHIVE DESTINATION lib
# )
