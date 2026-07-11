add_executable(cc3d_unit_volume
    test_volume_simulator.cpp
)

target_link_libraries(cc3d_unit_volume
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::VolumeTracker
        cc3d::Volume
)

gtest_discover_tests(cc3d_unit_volume
    TEST_PREFIX cc3d_unit_volume.
)

add_dependencies(cc3d_unit_all cc3d_unit_volume)
