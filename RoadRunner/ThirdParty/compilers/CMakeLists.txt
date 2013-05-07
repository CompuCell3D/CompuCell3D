cmake_minimum_required(VERSION 2.8)
project(compilers)
set(COMPILERS_FOLDER 			${CMAKE_CURRENT_SOURCE_DIR})


#=== TCC ==================================================
#FILE (GLOB tcc ${COMPILERS_FOLDER}/tcc/*.*)
#MESSAGE("TCC Glob: ${tcc}")

install (DIRECTORY ${COMPILERS_FOLDER}/tcc DESTINATION compilers COMPONENT rr_core)


