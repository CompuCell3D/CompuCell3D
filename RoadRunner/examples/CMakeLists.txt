cmake_minimum_required(VERSION 2.8)
PROJECT(rrExamples)
set(RR_INCLUDE_ROOT "../source")

# rr Includes
include_directories(${RR_INCLUDE_ROOT})
include_directories(${THIRD_PARTY_INSTALL_FOLDER}/include)
include_directories(${THIRD_PARTY_INSTALL_FOLDER}/include/sbml)
include_directories(${THIRD_PARTY_INSTALL_FOLDER}/include/cvode)

if(${MSVC})
endif(${MSVC})

set(examples 	
one_rr
one_thread
many_threads
load_model_dll
)

foreach(example ${examples})
 	add_subdirectory(${example})
	FILE (GLOB hdrs ${example}/*.h)
 	install (FILES ${hdrs} 						DESTINATION Examples/${example}	COMPONENT example_files)
	FILE (GLOB src ${example}/*.cpp)
 	install (FILES ${src} 						DESTINATION Examples/${example}	COMPONENT example_files)
 	install (FILES ${example}/Readme.txt 			DESTINATION Examples/${example}	COMPONENT example_files)
 	install (FILES ${example}/CMakeLists.txt 		DESTINATION Examples/${example}	COMPONENT example_files)
endforeach(example)

install (FILES Readme.txt 			DESTINATION Examples COMPONENT info)
install (FILES CMakeLists.txt 		DESTINATION Examples COMPONENT example_files)
