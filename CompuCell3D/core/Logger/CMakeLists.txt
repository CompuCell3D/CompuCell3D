
# C++ only executable - useful for debugging
OPTION(BUILD_LOGGER_DEMO "Build Logger Demo - debug compilation only" OFF)


set(SRCS
  Logger.cpp
  
)

include_directories (
    ${COMPUCELL3D_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core/Logger
)

if(WIN32 OR APPLE)
    add_static_library(LoggerLib ${SRCS})
else()
    add_shared_library(LoggerLib ${SRCS})
endif()

install_files(/include/CompuCell3D/Logger .h
    Logger
)


if (BUILD_LOGGER_DEMO)

    ADD_EXECUTABLE(LoggerDemo
        main.cpp
    )
    if (WIN32 OR APPLE)
        TARGET_LINK_LIBRARIES(LoggerDemo LoggerLibStatic)
    else()
        TARGET_LINK_LIBRARIES(LoggerDemo LoggerLibShared)
    endif()

    INSTALL_TARGETS(/bin LoggerDemo)


    # INCLUDE_DIRECTORIES (${EXPAT_INCLUDE_DIRS})
    # LINK_DIRECTORIES(${EXPAT_LIBRARY})

endif (BUILD_LOGGER_DEMO)




