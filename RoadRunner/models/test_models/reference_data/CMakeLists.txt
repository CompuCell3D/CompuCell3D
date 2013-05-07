cmake_minimum_required(VERSION 2.8)

PROJECT(ref_data)

set(ref_data_files
TestModel_1.dat
)

install(
FILES ${ref_data_files}
DESTINATION models/test_models/reference_data 	
COMPONENT testing
)
