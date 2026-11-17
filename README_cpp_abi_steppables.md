# Compiled Steppable ABI Demo

This demo mixes:

- one compiled C++ steppable built against `cc3d/kernel.h` only
- one ordinary Python steppable

The commands below are the exact sequence used on macOS in the `cc3d_compile` environment on August 29, 2026.

## Paths

```bash
export CC3D_REPO=/Users/m/src/conda-build-repos/CompuCell3D
export CC3D_SRC=$CC3D_REPO/CompuCell3D
export CC3D_ENV=/Users/m/miniconda3_arm64/envs/cc3d_compile
export CC3D_PYTHON=$CC3D_ENV/bin/python
export CC3D_CMAKE=$CC3D_ENV/bin/cmake
export CC3D_CLANG=$CC3D_ENV/bin/clang
export CC3D_CLANGXX=$CC3D_ENV/bin/clang++

export DEMO_DIR=$CC3D_SRC/core/Demos/SteppableDemos/CompiledSteppableABI
export DEMO_PROJECT=$DEMO_DIR/CompiledSteppableABI.cc3d
export DEMO_CPP=$DEMO_DIR/Native/GrowthSteppable.cpp
export DEMO_DYLIB=$DEMO_DIR/Native/GrowthSteppable.dylib
```

## 1. Activate the build environment

```bash
conda activate cc3d_compile
```

## 2. Rebuild and install CC3D with the new compiled-steppable support

This is required because `CompiledSteppable.cpp` is part of CC3D itself, not the standalone demo module.

```bash
cd $CC3D_SRC

export CC=$CC3D_CLANG
export CXX=$CC3D_CLANGXX
export PATH=$CC3D_ENV/bin:$PATH

mkdir -p build
cd build

$CC3D_CMAKE .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CC3D_CLANG \
  -DCMAKE_CXX_COMPILER=$CC3D_CLANGXX \
  -DPython3_EXECUTABLE=$CC3D_PYTHON \
  -DCMAKE_INSTALL_PREFIX=$CC3D_ENV

$CC3D_CMAKE --build . --parallel 8
$CC3D_CMAKE --install .
```

## 3. Compile the standalone demo steppable

You can either use the helper script below or run the equivalent manual command.

### Preferred: helper script

The helper script only builds the standalone compiled steppable. It does not rebuild CC3D itself.

```bash
cd $DEMO_DIR

$CC3D_PYTHON compile_steppable.py GrowthSteppable.cpp
```

Concrete command sequence verified on macOS on August 29, 2026:

```bash
python \
  -m cc3d.scripts.compile_steppable \
  /Users/m/src/conda-build-repos/CompuCell3D/CompuCell3D/core/Demos/SteppableDemos/CompiledSteppableABI/Native/GrowthSteppable.cpp \
  --repo /Users/m/src/conda-build-repos/CompuCell3D
```

To remove the compiled extension:

```bash
$CC3D_PYTHON compile_steppable.py GrowthSteppable.cpp --clean
```

Concrete cleanup command:

```bash
python \
  -m cc3d.scripts.compile_steppable \
  /Users/m/src/conda-build-repos/CompuCell3D/CompuCell3D/core/Demos/SteppableDemos/CompiledSteppableABI/Native/GrowthSteppable.cpp \
  --repo /Users/m/src/conda-build-repos/CompuCell3D \
  --clean
```

To point at a different checkout:

```bash
$CC3D_PYTHON compile_steppable.py GrowthSteppable.cpp --repo /path/to/CompuCell3D
```

### Manual command

On this machine, the conda `clang++` driver compiled the source but failed to link cleanly against Apple `ld`.
The command below uses Apple `clang++` for the standalone user module while still including only the public
`cc3d/kernel.h` header.

```bash
cd $DEMO_DIR

SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

/Library/Developer/CommandLineTools/usr/bin/clang++ \
  -O3 \
  -std=c++17 \
  -stdlib=libc++ \
  -dynamiclib \
  $DEMO_CPP \
  -I$CC3D_SRC/core \
  -isysroot "$SDKROOT" \
  -isystem "$SDKROOT/usr/include/c++/v1" \
  -o $DEMO_DYLIB
```

Verify:

```bash
ls -l $DEMO_DYLIB
file $DEMO_DYLIB
```

## 4. Run the demo

```bash
cd $CC3D_REPO

$CC3D_PYTHON -m cc3d.run_script \
  -i $DEMO_PROJECT
```

## 5. Expected behavior

You should see:

- `Compiled growth start`
- repeated `Compiled growth step mcs=... execution=...`
- `Python verifier start`
- every 10 MCS, a Python log line showing `cell_id`, `volume`, and updated `targetVolume`
- `Compiled growth finish`

The important check is that the Python steppable sees the `targetVolume` changes made by the compiled steppable
immediately after the compiled callback runs.

## Notes

- The demo resolves the dylib path from the `.cc3d` project location, not from `__file__`.
- Because `cc3d.run_script` uses `exec(code, globals(), locals())`, imports needed only inside helper functions
  should be done inside those functions.
- Warnings about `PlayerSizes`, `PlayerSizesFloating`, `RecentSimulations`, `MCSConversionFactor`, and
  `VoxelConversionFactor` are unrelated to the compiled-steppable prototype.
