add_library(cc3d_unit_fixtures STATIC
    CC3DTestSimulator.cpp
)

target_link_libraries(cc3d_unit_fixtures
    PUBLIC
        cc3d::CompuCellLib
        cc3d::XMLUtils
)

target_include_directories(cc3d_unit_fixtures
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
