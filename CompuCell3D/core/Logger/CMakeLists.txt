
# C++ only executable - useful for debugging
OPTION(BUILD_LOGGER_DEMO "Build Logger Demo - debug compilation only" OFF)


set(SRCS
  Logger.cpp
  
)

include_directories (
    ${COMPUCELL3D_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core/Logger
)

add_static_library(LoggerLib ${SRCS})

install_files(/include/CompuCell3D/Logger .h
    Logger
)


if (BUILD_LOGGER_DEMO)

    ADD_EXECUTABLE(LoggerDemo
        main.cpp
    )
    TARGET_LINK_LIBRARIES(LoggerDemo
        LoggerLibStatic
    )

    INSTALL_TARGETS(/bin LoggerDemo)


    # INCLUDE_DIRECTORIES (${EXPAT_INCLUDE_DIRS})
    # LINK_DIRECTORIES(${EXPAT_LIBRARY})

endif (BUILD_LOGGER_DEMO)




