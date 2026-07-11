add_executable(cc3d_unit_volume_tracker
    test_volume_tracker_simulator.cpp
)

target_link_libraries(cc3d_unit_volume_tracker
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::VolumeTracker
)

gtest_discover_tests(cc3d_unit_volume_tracker
    TEST_PREFIX cc3d_unit_volume_tracker.
)

add_dependencies(cc3d_unit_all cc3d_unit_volume_tracker)
