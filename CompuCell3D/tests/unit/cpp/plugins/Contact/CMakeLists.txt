add_executable(cc3d_unit_contact
    test_contact_simulator.cpp
)

target_link_libraries(cc3d_unit_contact
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::Contact
)

gtest_discover_tests(cc3d_unit_contact
    TEST_PREFIX cc3d_unit_contact.
)

add_dependencies(cc3d_unit_all cc3d_unit_contact)

add_executable(cc3d_unit_contact_hex
    test_contact_hex_simulator.cpp
)

target_link_libraries(cc3d_unit_contact_hex
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        GTest::gtest
        GTest::gtest_main
        cc3d::CellType
        cc3d::Contact
)

gtest_discover_tests(cc3d_unit_contact_hex
    TEST_PREFIX cc3d_unit_contact_hex.
)

add_dependencies(cc3d_unit_all cc3d_unit_contact_hex)
