# Draft C++ unit test layout.
#
# Proposed targets:
#   cc3d_unit_fixtures
#   cc3d_unit_contact
#   cc3d_unit_volume
#   cc3d_unit_surface
#
# Example future structure:
add_library(cc3d_unit_test_defaults INTERFACE)
target_compile_features(cc3d_unit_test_defaults INTERFACE cxx_std_17)
set_target_properties(cc3d_unit_test_defaults PROPERTIES
    CXX_EXTENSIONS OFF
)

add_custom_target(cc3d_unit_all)
add_custom_target(cc3d_conformance_all)

add_subdirectory(fixtures)
add_subdirectory(conformance)
add_subdirectory(plugins)
