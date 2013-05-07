CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_C_API_Examples)
#if(${MSVC})
#endif(${MSVC})

set(examples 	
#get_c_code
threads_1
threads_2
)


foreach(example ${examples})
 	add_subdirectory(${example})
	FILE (GLOB hdrs ${example}/*.h)
 	install (FILES ${hdrs} 						
                DESTINATION Examples/c_api/${example}	
                COMPONENT example_files)

	FILE (GLOB src ${example}/*.cpp)
 	install (FILES ${src} 						DESTINATION Examples/c_api/${example}	COMPONENT example_files)
 	install (FILES ${example}/Readme.txt 		DESTINATION Examples/c_api/${example}	COMPONENT example_files)
 	install (FILES ${example}/CMakeLists.txt 	DESTINATION Examples/c_api/${example}	COMPONENT example_files)
endforeach(example)

install (FILES Readme.txt 			DESTINATION Examples/c_api COMPONENT doc_file)
install (FILES CMakeLists.txt 		DESTINATION Examples/c_api COMPONENT example_files)
