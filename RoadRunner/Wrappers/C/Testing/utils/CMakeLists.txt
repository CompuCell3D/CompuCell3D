CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Tests)

configure_file(run_c_tests.sh.in run_c_tests.sh)

install(
FILES run_c_tests.bat 
DESTINATION bin
COMPONENT testing
)

install(
FILES ${CMAKE_CURRENT_BINARY_DIR}/run_c_tests.sh
DESTINATION bin
COMPONENT testing
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
)


