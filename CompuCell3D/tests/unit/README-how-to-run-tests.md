# How To Run Unit Tests

Be at the `CompuCell3D` repo root, not in `tests/unit` or the `Volume` test directory.

Use:

```bash
cd <CC3D_repo_root>/CompuCell3D
```

Example:

```bash
cd ~/src/conda-build-repos/CompuCell3D/CompuCell3D
```

Configure CMake from the repo root:

```bash
cmake -S . -B build-unit \
  -DCOMPUCELL3D_TEST=ON \
  -DBUILD_PYINTERFACE=OFF \
  -DBUILD_CPP_ONLY_EXECUTABLE=OFF \
  -DGTest_ROOT="$CONDA_PREFIX" \
  -DEIGEN3_INCLUDE_DIR="$PWD/core/Eigen"
```

Reason: the unit test target is added from the top-level `CMakeLists.txt`, and that configure step also pulls in the main CC3D libraries the test links against. Running `cmake` from `tests/unit` directly will not set up the full project correctly.

## Native Volume Unit Test

After configure, still from the repo root:

```bash
cmake --build build-unit --target cc3d_unit_volume -j4
ctest --test-dir build-unit -R cc3d_unit_volume --output-on-failure
```

## YAML Conformance Runner

The first portable interchange path is a YAML-driven conformance runner for deterministic local volume energy cases.

Additional prerequisite:

- `yaml-cpp` installed in the same environment used to configure and build CC3D

Build and run the conformance runner from the repo root:

```bash
cmake --build build-unit --target cc3d_conformance_runner -j4
ctest --test-dir build-unit -R cc3d_conformance_volume_single_pixel_gain_2d --output-on-failure
```

You can also run the executable directly on a specific YAML case:

```bash
build-unit/tests/unit/cpp/conformance/cc3d_conformance_runner \
  tests/unit/spec/cases/potts_core/volume_single_pixel_gain_2d.yaml
```

Current first-pass conformance support:

- schema version `1`
- domain `potts_local`
- fixed boundary conditions
- Moore neighborhood declarations
- `volume_energy_delta` queries only
