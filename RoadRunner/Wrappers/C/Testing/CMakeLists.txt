CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Tests)

set(target c_api_tests)

set(tests
Suite_Core_Tests
Suite_TestModel_1
Suite_SBML_Tests
)

add_executable( ${target} 
main.cpp
src/Args.cpp
src/TestUtils
src/rrSBMLTestSuiteSimulation_CAPI
${tests}
)

set_property(TARGET ${target}  
PROPERTY  COMPILE_DEFINITIONS 
STATIC_LIBSTRUCT
STATIC_RR
POCO_STATIC
XML_STATIC
POCO_NO_AUTOMATIC_LIBS
) 

if(UNIX)
set(staticLibPrefix ".a")
set(sharedLibPrefix ".so")
endif()

target_link_libraries (${target} 
rrc_api
unit_test-static
poco_xml-static
)

if(UNIX)
target_link_libraries (${target} 
lapack
blas
f2c
pthread
)
endif()

link_directories(
${THIRD_PARTY_INSTALL_FOLDER}/lib
)

include_directories(
"."
${RR_ROOT}
${THIRD_PARTY_FOLDER}/unit_test/src
${THIRD_PARTY_FOLDER}/rr-libstruct
)

add_subdirectory(utils)

install (TARGETS ${target}  		
DESTINATION bin COMPONENT testing
)




