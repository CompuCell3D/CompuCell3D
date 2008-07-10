SET(SRCS
        classes.cpp
)



ADD_LIBRARY(ClassesLib SHARED ${SRCS} )
TARGET_LINK_LIBRARIES(ClassesLib ${JAVA_JVM_LIBRARY})


add_executable(main main.cpp)
TARGET_LINK_LIBRARIES(main ${JAVA_JVM_LIBRARY})
INSTALL_TARGETS(/ main)




SET(LIBS
ClassesLib
)

SET(LIBS_AUX
)



INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}

)

SET(PACKAGE_NAME CalculatorTry)
SET(TARGET_SOURCE_PACKAGE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME})
message(${CMAKE_INSTALL_PREFIX})
message(${TARGET_SOURCE_PACKAGE_INSTALL_DIR})
file(MAKE_DIRECTORY ${TARGET_SOURCE_PACKAGE_INSTALL_DIR})

SET_SOURCE_FILES_PROPERTIES(classes.i  PROPERTIES CPLUSPLUS ON)
SET(CMAKE_SWIG_FLAGS -java -c++ -package ${PACKAGE_NAME} )

SET(CMAKE_SWIG_OUTDIR ${TARGET_SOURCE_PACKAGE_INSTALL_DIR} )

SWIG_ADD_MODULE(classes java classes.i)

message("REAL NAME=" ${SWIG_MODULE_classes_REAL_NAME})

# message("swig_generated_sources=" ${swig_generated_sources})



SWIG_LINK_LIBRARIES(classes ${LIBS} ${LIBS_AUX})
# SWIG_LINK_LIBRARIES(CompuCellAuxPython ${LIBS} ${LIBS_AUX})

install_targets(/CalculatorTry classes)

INSTALL_TARGETS(/${PACKAGE_NAME} RUNTIME_DIRECTORY /${PACKAGE_NAME}
  ClassesLib)



#installs while cmake is run
# file(GLOB dummy_java_files "@CMAKE_CURRENT_SOURCE_DIR@/*.java")
# file(INSTALL DESTINATION "@CMAKE_BINARY_DIR@/jni_test/" TYPE FILE FILES ${dummy_java_files})


# file(GLOB java_files "@CMAKE_BINARY_DIR@/jni_test/*.java")
#
#
# # file(INSTALL DESTINATION "@CMAKE_INSTALL_PREFIX@/java" TYPE FILE FILES ${java_files})
#
# #installs during install
# INSTALL(FILES ${java_files} DESTINATION "@CMAKE_INSTALL_PREFIX@/java" )

#  INSTALL_FILES(/java ${CMAKE_BINARY_DIR}/jni_test/.*\\.java$)

# INSTALL_FILES(/java FILES *.java ${CMAKE_BINARY_DIR}/jni_test/)




# install(
#    CODE "
#    file(GLOB java_files ${CMAKE_BINARY_DIR}/jni_test/CalculatorTry/*.java)
#    message(\${java_files})
#    message(\${CMAKE_INSTALL_PREFIX})
#    file(INSTALL DESTINATION \${CMAKE_INSTALL_PREFIX}/CalculatorTry TYPE FILE FILES \${java_files})
#    "
# )


# set(java_files_path ${CMAKE_BINARY_DIR}/jinterface/jni_test)
#
# INSTALL(FILES ${java_files_path}/*.java
#        DESTINATION /java)



