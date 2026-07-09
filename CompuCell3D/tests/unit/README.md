# CC3D Unit Test Architecture Draft

This directory is the proposed home for deterministic unit and conformance tests.
It complements the existing simulation/regression tests in `tests/` and does not replace them.

## Goals

- Add native unit tests for plugin-local logic.
- Add a shared YAML-based conformance suite that can be reused by other Cellular Potts Model implementations.
- Keep deterministic local-energy tests separate from full simulation regression tests.

## Directory Layout

```text
tests/
  unit/
    README.md
    CMakeLists.txt
    cpp/
      CMakeLists.txt
      fixtures/
        CMakeLists.txt
        README.md
      plugins/
        CMakeLists.txt
        Contact/
          CMakeLists.txt
        Volume/
          CMakeLists.txt
        Surface/
          CMakeLists.txt
    spec/
      README.md
      schema/
        potts_unit_test_v1.yaml
      cases/
        potts_core/
          contact_basic_interface_2d.yaml
          contact_medium_exposure_2d.yaml
          contact_periodic_wrap_2d.yaml
          volume_single_pixel_gain_2d.yaml
          surface_single_pixel_gain_2d.yaml
    python/
      README.md
```

## Test Layers

1. Native unit tests
   These are C++ tests for deterministic plugin logic and local energy calculations.

2. Component tests
   These use a lightweight CC3D fixture to construct a tiny lattice, cells, and plugin state without running a full simulation.

3. Regression tests
   These remain in the existing `tests/` area and execute full CC3D runs.

## Running Native Tests

Prerequisites:

- a working C++ compiler toolchain for your platform
- `cmake`
- `ctest`
- `gtest` installed in the active build environment
- a `gtest` build compatible with C++17; the native unit-test targets are compiled as C++17 even though the main CC3D code still builds with its existing default

Recommended approach:

- use the same environment that you normally use to compile CC3D
- for native unit tests only, configure with `-DBUILD_PYINTERFACE=OFF`

### macOS and Linux

From the repo root:

```bash
cmake -S . -B build-unit \
  -DCOMPUCELL3D_TEST=ON \
  -DBUILD_PYINTERFACE=OFF \
  -DBUILD_CPP_ONLY_EXECUTABLE=OFF \
  -DGTest_ROOT="$CONDA_PREFIX" \
  -DEIGEN3_INCLUDE_DIR="$PWD/core/Eigen"

cmake --build build-unit --target cc3d_unit_volume -j4

ctest --test-dir build-unit -R cc3d_unit_volume --output-on-failure
```

If `cmake` and `ctest` are installed only inside a conda environment, activate that environment first.
If `gtest` is also coming from conda, keep `GTest_ROOT` pointed at the active environment to avoid mixing headers from one environment with libraries from another.

### Windows

Run from a developer shell with your compiler environment enabled and the CC3D build environment activated:

```bat
cmake -S . -B build-unit ^
  -DCOMPUCELL3D_TEST=ON ^
  -DBUILD_PYINTERFACE=OFF ^
  -DBUILD_CPP_ONLY_EXECUTABLE=OFF ^
  -DGTest_ROOT=%CONDA_PREFIX% ^
  -DEIGEN3_INCLUDE_DIR=%CD%/core/Eigen

cmake --build build-unit --target cc3d_unit_volume --config Release

ctest --test-dir build-unit -R cc3d_unit_volume --output-on-failure -C Release
```

### Notes

- `EIGEN3_INCLUDE_DIR` is pointed at the vendored Eigen tree in this repo to avoid picking up an incompatible external Eigen configuration.
- The current native test target links the `CellType`, `VolumeTracker`, and `Volume` plugins directly. That means plugin registration happens through normal plugin proxy initialization and does not require setting `COMPUCELL3D_PLUGIN_PATH`.
- The first implemented native test is `cc3d_unit_volume`, which exercises `VolumePlugin::changeEnergy(...)` using a real `Simulator`, `Potts3D`, and cell field.

### Running Under Codex

If you are running the build through the Codex sandbox used in this session, CMake also writes `../cc3d/_version.py` during configure. That path is outside the writable workspace root for this repo checkout, so configure may require an approval for unsandboxed execution. In a normal local shell this is not a special issue.

## CMake Target Structure

Recommended tools:

- `GoogleTest` for native C++ tests
- `CTest` for build integration
- `pytest` for YAML-driven orchestration and cross-implementation runners

Recommended target layout:

- `cc3d_unit_fixtures`
  Shared helper library for constructing tiny deterministic CPM states.

- `cc3d_unit_contact`
  Native tests for Contact plugin behavior.

- `cc3d_unit_volume`
  Native tests for Volume plugin behavior.

- `cc3d_unit_surface`
  Native tests for Surface plugin behavior.

- `cc3d_conformance_runner`
  A lightweight executable or Python entry point that loads YAML cases and dispatches them to the fixture layer.

The intended dependency direction is:

```text
plugin test target -> cc3d_unit_fixtures -> CC3D core libraries
YAML conformance runner -> cc3d_unit_fixtures -> CC3D core libraries
```

The fixture library should avoid linking against UI or full application startup code.

## Fixture Design

The fixture layer should expose a small, stable API for constructing local CPM test states.
It should be explicit and simulator-light.

Suggested responsibilities:

- Create a lattice with a specified `Dim3D`
- Install boundary conditions
- Create cells with IDs and types
- Paint pixels/voxels into the cell field
- Configure neighborhood order or explicit neighbor definition
- Configure plugin parameters for the unit under test
- Execute deterministic local queries

Suggested query surface:

- `contact_energy_delta(source_pt, target_pt)`
- `volume_energy_delta(source_cell_id, target_cell_id)`
- `surface_energy_delta(source_pt, target_pt)`
- `cell_id_at(pt)`
- `cell_type_at(pt)`

Suggested fixture classes:

- `PottsTestLattice`
  Owns lattice dimensions, boundary conditions, and the cell field.

- `PottsTestCells`
  Creates cells, types, and pixel occupancy.

- `PottsEnergyFixture`
  Wires a minimal `Potts3D` plus selected plugins and exposes local energy queries.

- `YamlConformanceCase`
  In-memory representation of one portable test case.

The first version should support only deterministic local queries. It should not attempt full MCS stepping or stochastic acceptance.

## YAML Schema Draft

The shared schema should describe observable CPM semantics, not CC3D internals.

Top-level fields:

- `version`
- `name`
- `domain`
- `lattice`
- `cells`
- `medium`
- `parameters`
- `queries`

Minimal draft:

```yaml
version: 1
name: contact-basic-interface-2d
domain: potts_local
lattice:
  dim: [6, 4, 1]
  boundary:
    x: fixed
    y: fixed
    z: fixed
  neighborhood:
    kind: moore
    order: 1
cells:
  - id: 1
    type: A
    pixels:
      - [1, 1, 0]
      - [1, 2, 0]
  - id: 2
    type: B
    pixels:
      - [2, 1, 0]
      - [2, 2, 0]
medium:
  type: Medium
parameters:
  contact:
    A: {A: 10, B: 14, Medium: 16}
    B: {A: 14, B: 12, Medium: 18}
    Medium: {A: 16, B: 18, Medium: 0}
queries:
  - id: q1
    kind: contact_energy_delta
    source: [1, 1, 0]
    target: [2, 1, 0]
    expected:
      value: 4.0
      tolerance: 1.0e-12
```

Schema constraints for v1:

- Cases must be deterministic.
- All expected outputs must be numeric or exact symbolic values.
- Queries must refer only to lattice positions and declared cells.
- No simulator-stepping loops.
- No random seeds in v1.

## First 5 Pilot Test Cases

1. `contact_basic_interface_2d.yaml`
   Two adjacent cells of different types on a fixed 2D lattice. Validate one local contact-energy delta.

2. `contact_medium_exposure_2d.yaml`
   One cell adjacent to medium. Validate that medium-specific contact energies are used correctly.

3. `contact_periodic_wrap_2d.yaml`
   Two cells interacting across a periodic boundary. Validate that the neighborhood logic wraps correctly.

4. `volume_single_pixel_gain_2d.yaml`
   One candidate copy increases source/target volume by one pixel. Validate the local volume constraint contribution.

5. `surface_single_pixel_gain_2d.yaml`
   One candidate copy changes local perimeter/surface. Validate the local surface constraint contribution.

## Implementation Order

1. Add `GoogleTest` and `CTest` plumbing under `tests/unit/cpp`.
2. Build the `cc3d_unit_fixtures` helper library.
3. Implement native Contact tests first.
4. Implement a YAML loader for the v1 schema.
5. Make the first three Contact cases executable from both C++ and Python-driven runners.
6. Add Volume and Surface component tests after the fixture API stabilizes.

## Non-Goals For V1

- Full MCS trajectory validation
- Stochastic acceptance behavior
- Multi-plugin coupled scenarios
- Performance benchmarking
- GUI or Player integration
