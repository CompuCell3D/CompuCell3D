add_executable(cc3d_unit_surface
    test_surface_simulator.cpp
)

target_link_libraries(cc3d_unit_surface
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::SurfaceTracker
        cc3d::Surface
)

gtest_discover_tests(cc3d_unit_surface
    TEST_PREFIX cc3d_unit_surface.
)

add_dependencies(cc3d_unit_all cc3d_unit_surface)
