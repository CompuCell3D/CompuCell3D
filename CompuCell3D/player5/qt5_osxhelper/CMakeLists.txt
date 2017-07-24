cmake_minimum_required(VERSION 2.8)
if(POLICY CMP0053)
  cmake_policy(SET CMP0053 NEW)
endif()

find_library( LIB_COCOA cocoa )

add_library(osxhelper SHARED osxHelper.mm)
target_link_libraries(osxhelper ${LIB_COCOA})

add_library(cpposxhelper SHARED cpp_osx_helper.cpp osxHelper.mm)
target_link_libraries(cpposxhelper ${osxhelper} ${LIB_COCOA})
install_targets(/player5/Utilities cpposxhelper)
