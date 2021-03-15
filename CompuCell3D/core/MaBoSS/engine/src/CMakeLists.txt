macro(MABOSS_GEN source_file target_file label)
    # Read the source file, make replacements and output target file
    file(READ ${source_file} source_string)
    string(REPLACE "yy" ${label} s "${source_string}")
    string(REPLACE "register" "" target_string "${s}")
    file(WRITE ${target_file} "${target_string}")
endmacro(MABOSS_GEN)

SET(MaBoSS_SRCS 
    BooleanGrammar.cc 
    BooleanNetwork.cc 
    BuiltinFunctions.cc 
    Cumulator.cc 
    EnsembleEngine.cc 
    FinalStateDisplayer.cc 
    FinalStateSimulationEngine.cc 
    FixedPointDisplayer.cc 
    Function.cc 
    LogicalExprGen.cc 
    MaBEstEngine.cc 
    MBDynBitset.cc 
    MetaEngine.cc 
    ProbaDist.cc 
    ProbTrajDisplayer.cc 
    RunConfig.cc 
    RunConfigGrammar.cc 
    StatDistDisplayer.cc 
    StochasticSimulationEngine.cc 
    user_func.cc 
    Utils.cc 
    maboss-config.cpp
)
if(MINGW)
    list(APPEND MaBoSS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../extern/sys/times.cpp)
endif()

SET(MaBoSS_HDRS
    BooleanGrammar.h 
    BooleanNetwork.h 
    Client.h 
    Cumulator.h 
    DataStreamer.h 
    EnsembleEngine.h 
    FinalStateDisplayer.h 
    FinalStateSimulationEngine.h 
    FixedPointDisplayer.h 
    Function.h 
    MaBEstEngine.h 
    MBDynBitset.h 
    MetaEngine.h 
    ProbaDist.h 
    Probe.h 
    ProbTrajDisplayer.h 
    RandomGenerator.h 
    RPC.h 
    RunConfig.h 
    Server.h 
    StatDistDisplayer.h 
    StochasticSimulationEngine.h 
    Utils.h 
    maboss-config.h
)

set(MaBoSS_AUXS 
    BooleanGrammar.l
    BooleanGrammar.y
    RunConfigGrammar.l
    RunConfigGrammar.y
)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

####################
# Generate scripts #
####################

# BooleanGrammar.cc
execute_process(
    COMMAND ${FLEX_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/BooleanGrammar.l"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
execute_process(COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_BINARY_DIR}/lex.yy.c" "${CMAKE_BINARY_DIR}/lex.CTBNDL.c")
MABOSS_GEN("${CMAKE_BINARY_DIR}/lex.CTBNDL.c" "${CMAKE_CURRENT_SOURCE_DIR}/lex.CTBNDL.cc" "CTBNDL")

execute_process(
    COMMAND ${BISON_EXECUTABLE} -v "${CMAKE_CURRENT_SOURCE_DIR}/BooleanGrammar.y"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
MABOSS_GEN("${CMAKE_BINARY_DIR}/BooleanGrammar.tab.c" "${CMAKE_CURRENT_SOURCE_DIR}/BooleanGrammar.cc" "CTBNDL")

file(COPY "${CMAKE_BINARY_DIR}/BooleanGrammar.tab.c" DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# RunConfigGrammar.cc
execute_process(
    COMMAND ${FLEX_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/RunConfigGrammar.l"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
execute_process(COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_BINARY_DIR}/lex.yy.c" "${CMAKE_BINARY_DIR}/lex.RC.c")
MABOSS_GEN("${CMAKE_BINARY_DIR}/lex.RC.c" "${CMAKE_CURRENT_SOURCE_DIR}/lex.RC.cc" "RC")

execute_process(
    COMMAND ${BISON_EXECUTABLE} -v "${CMAKE_CURRENT_SOURCE_DIR}/RunConfigGrammar.y"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
MABOSS_GEN("${CMAKE_BINARY_DIR}/RunConfigGrammar.tab.c" "${CMAKE_CURRENT_SOURCE_DIR}/RunConfigGrammar.cc" "RC")

file(COPY "${CMAKE_BINARY_DIR}/RunConfigGrammar.tab.c" DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

#################
# Build library #
#################

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows AND NOT MINGW)
    add_library(MaBoSSShared SHARED IMPORTED)
    set_target_properties(MaBoSSShared PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../win/bin/libMaBoSSShared.dll 
        IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/libMaBoSSShared.lib
    )

else()
    set(CMAKE_HAVE_LIBC_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    set(MABOSS_LIBS Threads::Threads)

    if(MINGW)
        add_library(LIBGCC_IMPORTED UNKNOWN IMPORTED GLOBAL)
        set_target_properties(LIBGCC_IMPORTED PROPERTIES 
            IMPORTED_LOCATION ${Python_INCLUDE_DIRS}/../Library/mingw-w64/bin/libgcc_s_seh-1.dll 
            LINKER_LANGUAGE C
        )
        add_library(LIBDL_IMPORTED UNKNOWN IMPORTED GLOBAL)
        set_target_properties(LIBDL_IMPORTED PROPERTIES 
            IMPORTED_LOCATION ${Python_INCLUDE_DIRS}/../Library/mingw-w64/bin/libdl.dll 
            LINKER_LANGUAGE C
        )
        list(APPEND MABOSS_LIBS LIBGCC_IMPORTED LIBDL_IMPORTED)
    endif()

    ADD_SHARED_LIBRARY(MaBoSS ${MaBoSS_SRCS} 
    LINK_LIBRARIES ${MABOSS_LIBS})
    target_compile_options(MaBoSSShared PUBLIC -DMAXNODES=${MABOSS_MAXNODES})

    if(MINGW)
        target_include_directories(MaBoSSShared PUBLIC 
            ${CMAKE_CURRENT_SOURCE_DIR}/../extern 
            ${Python_INCLUDE_DIRS}/../Library/mingw-w64/include
        )
    endif()

endif()

###########
# Install #
###########
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL Windows OR MINGW)
    install_targets(${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR} MaBoSSShared)
else()
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/../win/bin/libMaBoSSShared.dll 
        DESTINATION ${COMPUCELL3D_INSTALL_SWIG_MODULES_DIR}/bin
    )
    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/libMaBoSSShared.lib 
        DESTINATION ${COMPUCELL3D_INSTALL_LIB_DIR}
    )
endif()
install(FILES ${MaBoSS_HDRS} DESTINATION include/CompuCell3D/MaBoSS/engine/src)

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../extern/sys/times.h DESTINATION include/CompuCell3D/MaBoSS/extern/sys)
endif()
