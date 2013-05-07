CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(Apps)
set(RR_INCLUDE_ROOT "../source")

# rr Includes
include_directories(
${RR_INCLUDE_ROOT}
${THIRD_PARTY_INSTALL_FOLDER}/include
${THIRD_PARTY_INSTALL_FOLDER}/include/sbml
${THIRD_PARTY_INSTALL_FOLDER}/include/cvode
${THIRD_PARTY_INSTALL_FOLDER}/include/clapack
)

set(apps 	
		rr
#         rr_test_suite_tester
#        rr_performance_tester
)

set(app_dir Apps/cpp)

foreach(app ${apps})
 	add_subdirectory(${app})
#	FILE (GLOB hdrs ${app}/*.h)
# 	install (FILES ${hdrs} 						DESTINATION ${app_dir}/${app}	COMPONENT example_files)
#	FILE (GLOB source ${app}/*.cpp)
# 	install (FILES ${source} 					DESTINATION ${app_dir}/${app}	COMPONENT example_files)
# 	install (FILES ${app}/Readme.txt 			DESTINATION ${app_dir}/${app}	COMPONENT example_files)
# 	install (FILES ${app}/CMakeLists.txt 		DESTINATION ${app_dir}/${app}	COMPONENT example_files)
endforeach(app)
#
#install (FILES Readme.txt 			DESTINATION ${app_dir} COMPONENT info)
#install (FILES CMakeLists.txt 		DESTINATION ${app_dir} COMPONENT example_files)
