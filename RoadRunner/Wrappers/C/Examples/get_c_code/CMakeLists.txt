set(target get_c_code)

set(rrSrc "../../../../source")
add_executable( ${target} 
main.cpp
${rrSrc}/rrFileLog.cpp
${rrSrc}/rrLogger.cpp
${rrSrc}/rrLogFile.cpp
${rrSrc}/rrLoggerUtils.cpp
${rrSrc}/rrLogOutput.cpp
${rrSrc}/rrObject.cpp
${rrSrc}/rrStringUtils.cpp
${rrSrc}/rrUtils.cpp
)

set_property(   TARGET ${target} 
                PROPERTY  COMPILE_DEFINITIONS 
                          STATIC_RR             #when linking with static lib, no declspec shuld be present..
)
target_link_libraries (${target} rrc_api roadrunner)

install(
        TARGETS ${target} 		
        DESTINATION bin 
        COMPONENT compiled_examples)
