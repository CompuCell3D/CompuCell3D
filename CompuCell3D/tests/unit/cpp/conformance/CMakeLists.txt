find_package(yaml-cpp CONFIG REQUIRED)

add_executable(cc3d_conformance_runner
    cc3d_conformance_runner.cpp
    YamlConformanceLoader.cpp
)

target_link_libraries(cc3d_conformance_runner
    PRIVATE
        cc3d_unit_test_defaults
        cc3d_unit_fixtures
        cc3d::CellType
        cc3d::VolumeTracker
        cc3d::Volume
)

if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(cc3d_conformance_runner PRIVATE yaml-cpp::yaml-cpp)
else()
    target_link_libraries(cc3d_conformance_runner PRIVATE yaml-cpp)
endif()

target_include_directories(cc3d_conformance_runner
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
    NAME cc3d_conformance_volume_single_pixel_gain_2d
    COMMAND
        $<TARGET_FILE:cc3d_conformance_runner>
        ${CMAKE_SOURCE_DIR}/tests/unit/spec/cases/potts_core/volume_single_pixel_gain_2d.yaml
)

add_dependencies(cc3d_conformance_all cc3d_conformance_runner)
