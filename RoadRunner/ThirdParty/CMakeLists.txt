cmake_minimum_required(VERSION 2.8)
project(RRThirdPartyBuild)

#----- ThirdParties
set(THIRD_PARTY_FOLDER 			${CMAKE_CURRENT_SOURCE_DIR})

#--- mark pre-set thirdparty flags as advanced
mark_as_advanced(
WITH_BZIP2        
WITH_SWIG         
WITH_ZLIB         
WITH_CPP_NAMESPACE
BUILD_CVODE   
BUILD_CVODES   
BUILD_IDA      
BUILD_IDAS     
BUILD_KINSOL   
BUILD_UNIT_TEST
BUILD_SHARED_LIBS
BUILD_STATIC_LIBS
ENABLE_LAYOUT
EXAMPLES_ENABLE
EXTRA_LIBS
FCMIX_ENABLE
LAPACK_ENABLE
LIBSBML_DEPENDENCY_DIR
LIBSTRUCT_BUILD_SHARED_LIBS
LIBSTRUCT_BUILD_STATIC_LIBS
LIBXML_INCLUDE_DIR
LIBZ_LIBRARY
LIBBZ_LIBRARY
LIBXML_LIBRARY
MPI_ENABLE
NLEQ_BUILD_SHARED_LIBS
NLEQ_BUILD_STATIC_LIBS
POCO_BUILD_SHARED_LIBS
POCO_BUILD_STATIC_LIBS
SUNDIALS_PRECISION
WITH_CHECK
WITH_CSHARP
WITH_DOXYGEN
WITH_EXAMPLES
WITH_EXPAT
WITH_JAVA
WITH_LIBXML
WITH_MATLAB
WITH_OCTAVE
WITH_PERL
WITH_PYTHON
WITH_R
WITH_RUBY
WITH_STATIC_RUNTIME
WITH_XERCES
)
#pre set libsbml options
set(WITH_BZIP2          OFF CACHE BOOL "test")
set(WITH_SWIG           OFF CACHE BOOL "test")
set(WITH_ZLIB           OFF CACHE BOOL "test")
set(WITH_CPP_NAMESPACE  ON  CACHE BOOL "test")

set(BUILD_CVODES        OFF CACHE BOOL "")
set(BUILD_IDA           OFF CACHE BOOL "")
set(BUILD_IDAS          OFF CACHE BOOL "")
set(BUILD_KINSOL        OFF CACHE BOOL "")
set(BUILD_UNIT_TEST     OFF CACHE BOOL "")

#LIBXML stuff
set(LIBXML_INCLUDE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libsbml/include/" CACHE string "test")

if(${BORLAND})
set(FOLDER_POSTFIX "xe")
set(LIBXML_LIBRARY      "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libsbml/lib/libxml2_xe.lib" CACHE string "test")
endif(${BORLAND})

if(${MSVC})
set(FOLDER_POSTFIX "vs")
set(LIBXML_LIBRARY      "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libsbml/lib/libxml2.lib" CACHE string "test")
endif()

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "Choose the build type. The options are: None (CMAKE_CXX_FLAGS or
     CMAKE_C_FLAGS are used), Debug, Release, RelWithDebInfo, MinSizeRel.")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
  "Choose the build type. The options are: None (CMAKE_CXX_FLAGS or
  CMAKE_C_FLAGS are used), Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

set(exe_path		 			${PROJECT_BINARY_DIR}/bin)
set(lib_path		 			${PROJECT_BINARY_DIR}/lib)

IF(${MSVC})
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(exe_path ${exe_path}/Debug)
        set(lib_path ${lib_path}/Debug)
    ELSE(CMAKE_BUILD_TYPE MATCHES "Debug")
           set(exe_path ${exe_path}/Release)
           set(lib_path ${lib_path}/Release)
    ENDIF(CMAKE_BUILD_TYPE MATCHES "Debug")
ENDIF(${MSVC})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY 	${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY 	${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY 	${PROJECT_BINARY_DIR}/bin)

link_directories( 
${THIRD_PARTY_FOLDER}/dependencies/libsbml/lib
)

#=== COMPILER FLAGS 

# UNIX matches ALL Unicies, including Apple (from Andy S.)
if(${UNIX})
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  SET (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fPIC")
endif()

# Only matches Linux, Linux has some system headers that conflict with std::isnan
# when building libsbml
if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++0x")
endif()

#if(${MSVC})
#SET (CMAKE_C_FLAGS_RELEASE 			"/MD /Od /Ob0 /D NDEBUG /P")
#endif()

if(${BORLAND})
add_definitions(
             -w-8012                     #Comparing signed /unsigned
             -w-8057                     #Parameter never used
             -w-8004                     #'var' is assigned a value that is never used
             -w-8123                     #Path not found
             -w-8066                     #Unreachable code
             -w-8104                     #Local static with constructor...
             -w-8008                     #Condition is always false..
        )
endif()

if(${MSVC})
add_definitions(
-D_CRT_SECURE_NO_WARNINGS 
-D_CRT_NONSTDC_NO_DEPRECATE
"-wd4244"
"-wd4554" #use paranthesis to clarify precedence
)
endif()

#Add Third Party Folders
add_subdirectory(libsbml)
add_subdirectory(clapack/3.2.1)
add_subdirectory(sundials)
add_subdirectory(pugi)
add_subdirectory(nleq)
add_subdirectory(rr-libstruct)
add_subdirectory(unit_test)
add_subdirectory(compilers)
add_subdirectory(poco)
add_subdirectory(rr_support)

#==== SYSTEM FILES (COMPILER SPECIFICS) =================================================
if(${BORLAND})
set(CG_RUNTIMES		cc32110MT.dll)
foreach(runtime ${CG_RUNTIMES})
install (FILES 
            ${THIRD_PARTY_FOLDER}/dependencies/cg/xe/${runtime} 
            DESTINATION bin 
            COMPONENT rr_core)
endforeach()
install (FILES 
            ${THIRD_PARTY_FOLDER}/dependencies/libsbml/lib/libxml2_xe.lib 
            DESTINATION lib 
            COMPONENT rr_core)
endif()

if(WIN32)
    set(RUNTIMES libxml2.DLL iconv.dll zlib1.dll)
    foreach(runtime ${RUNTIMES})
    install(    FILES 	${THIRD_PARTY_FOLDER}/dependencies/libsbml/bin/${runtime} 
                DESTINATION bin 
                COMPONENT rr_core)
    endforeach()

endif()
