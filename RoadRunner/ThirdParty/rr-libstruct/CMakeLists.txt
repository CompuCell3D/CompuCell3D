CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(rr-libstruct)

# Option that allows users to build static and/or shared libraries
# ----------------------------------------------------------------
OPTION(LIBSTRUCT_BUILD_STATIC_LIBS "Build static libraries" ON)
OPTION(LIBSTRUCT_BUILD_SHARED_LIBS "Build shared libraries" ON)
set(target rr-libstruct) 

set(sources 
lsMatrix
lsComplex
lsLibStructural
lsSBMLModel
lsUtils
lsLibla
lsLA
lsLUResult
)


include_directories(
.
${THIRD_PARTY_FOLDER}/clapack/3.2.1/INCLUDE
${THIRD_PARTY_FOLDER}/libsbml/src
)

#add_library(${target} SHARED ${sources})
#
#set_property(TARGET ${target}  
#             PROPERTY  COMPILE_DEFINITIONS 
#                       WIN32
#                       SHARED_NLEQ
#             ) 
#

if (${LIBSTRUCT_BUILD_STATIC_LIBS})
# static version
    add_library(${target}-static STATIC ${sources})

    target_link_libraries (${target}-static    
    f2c
    blas
    lapack

    )

    set(static_compile_definitions
    STATIC_LIBSTRUCT
    LIBSBML_USE_CPP_NAMESPACE
    LIBSBML_STATIC
    )

    # if(WIN32)
    # set(static_compile_definitions 
    # ${static_compile_definitions}
    # WIN32
    # ) 

    set_property(TARGET ${target}-static  
             PROPERTY  COMPILE_DEFINITIONS 
             ${static_compile_definitions}
             ) 

install(TARGETS ${target}-static 
        LIBRARY DESTINATION lib 
        ARCHIVE DESTINATION lib)
             
    
endif()


if (${LIBSTRUCT_BUILD_SHARED_LIBS})

    set(shared_compile_definitions
    SHARED_LIBSTRUCT
    LIBSBML_USE_CPP_NAMESPACE
    LIBSBML_SHARED
    )


    add_library(${target} SHARED ${sources})

    SET(EXTRALIBS sbml5)
    
    if (WIN32)
        SET(EXTRALIBS libsbml5)
    endif()
    
    
    target_link_libraries (${target}
     ${EXTRALIBS}
     f2c
     blas
     lapack

    )
    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows) 
        set(shared_compile_definitions 
        ${shared_compile_definitions}
        WIN32
        ) 
    else(${CMAKE_SYSTEM_NAME} STREQUAL Windows) 
        set(shared_compile_definitions 
        ${shared_compile_definitions}
        ) 
    
    endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows) 
    
    set_property(TARGET ${target}  
                 PROPERTY  COMPILE_DEFINITIONS 
                 ${shared_compile_definitions}
                 ) 
                 

    if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)                 
        install(TARGETS ${target} 
                RUNTIME DESTINATION bin
                LIBRARY DESTINATION lib 
                ARCHIVE DESTINATION lib)
     else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)             
     
        install(TARGETS ${target} 
                
                LIBRARY DESTINATION lib 
                ARCHIVE DESTINATION lib)     
     endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)       

endif()
             

# if(WIN32)
# set(shared_compile_definitions 
# ${shared_compile_definitions}
# WIN32
# ) 
# endif()

             
# #libraries to link with
# if(${MINGW})
# else()
# #target_link_libraries (${target} libf2c)
# #target_link_libraries (${target}-static libf2c)
# endif()

set(headers 
lsExporter.h
lsMatrix.h
lsComplex.h
lsLibStructural.h
lsLA.h
lsLibla.h
lsLUResult.h
)

#Install required files
install(FILES ${headers} 
        DESTINATION include/rr-libstruct)

#install(TARGETS ${target} 
#        LIBRARY DESTINATION lib 
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin)
#

