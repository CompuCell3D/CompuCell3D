cmake_minimum_required(VERSION 2.8)

PROJECT(rrModels)

set(test_models
TestModel_1.xml
)

if(BUILD_TESTS)
add_subdirectory(reference_data)
endif()

install(    
FILES ${test_models}
DESTINATION models/test_models 	
COMPONENT testing
)
