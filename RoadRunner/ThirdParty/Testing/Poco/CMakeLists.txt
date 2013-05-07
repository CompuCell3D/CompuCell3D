cmake_minimum_required(VERSION 2.8)


include_directories(
${THIRD_PARTY_INSTALL_FOLDER}/include
    
)

link_directories(
${THIRD_PARTY_INSTALL_FOLDER}/lib
)


add_subdirectory(SharedLibraries)

#add_subdirectory(Threads)
