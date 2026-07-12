add_executable(cc3d_unit_surface_tracker
    test_surface_tracker_simulator.cpp
)

target_link_libraries(cc3d_unit_surface_tracker
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::SurfaceTracker
)

gtest_discover_tests(cc3d_unit_surface_tracker
    TEST_PREFIX cc3d_unit_surface_tracker.
)

add_dependencies(cc3d_unit_all cc3d_unit_surface_tracker)

add_executable(cc3d_unit_surface_tracker_hex
    test_surface_tracker_hex_simulator.cpp
)

target_link_libraries(cc3d_unit_surface_tracker_hex
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::SurfaceTracker
)

gtest_discover_tests(cc3d_unit_surface_tracker_hex
    TEST_PREFIX cc3d_unit_surface_tracker_hex.
)

add_dependencies(cc3d_unit_all cc3d_unit_surface_tracker_hex)
