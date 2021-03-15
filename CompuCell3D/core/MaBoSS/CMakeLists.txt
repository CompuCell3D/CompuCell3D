if(NOT MABOSS_MAXNODES)
    set(MABOSS_MAXNODES 64)
endif()

add_subdirectory(engine)

INCLUDE_DIRECTORIES(
    ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS
    ${COMPUCELL3D_SOURCE_DIR}/core/MaBoSS/engine/src
)

set(CC3DMaBoSS_SRCS
    CC3DMaBoSS.cpp
)
set(CC3DMaBoSS_HDRS
    CC3DMaBoSS.h
    CC3DMaBoSSDLLSpecifier.h
)

#################
# Build library #
#################

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND NOT MINGW)
    add_library(MaBoSSCC3DShared SHARED IMPORTED)
    set_target_properties(MaBoSSCC3DShared PROPERTIES
        IMPORTED_LOCATION engine/win/bin/libMaBoSSCC3DShared.dll
        IMPORTED_IMPLIB engine/win/lib/libMaBoSSCC3DShared.lib
    )
else()
    ADD_SHARED_LIBRARY(MaBoSSCC3D ${CC3DMaBoSS_SRCS} 
        LINK_LIBRARIES 
        MaBoSSShared
    )
    target_compile_options(MaBoSSCC3DShared PUBLIC -DMAXNODES=${MABOSS_MAXNODES})
endif()

###########
# Install #
###########

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND NOT MINGW)
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/engine/win/bin/libMaBoSSCC3DShared.dll 
        DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/bin
    )
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/engine/win/lib/libMaBoSSCC3DShared.lib 
        DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
    )
else()
    install_targets(${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} MaBoSSCC3DShared)
endif()

install(FILES ${CC3DMaBoSS_HDRS} DESTINATION include/CompuCell3D/MaBoSS)
