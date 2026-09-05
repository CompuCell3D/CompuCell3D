# Compiling Native Steppables

`compile_steppable.py` builds a single C++ steppable that uses the public `cc3d/kernel.h` ABI. The output is written next to the source file:

- Windows: `.dll`
- macOS: `.dylib`
- Linux: `.so`

Run the command from an activated CC3D conda environment so `python -m cc3d.scripts.compile_steppable` resolves to the installed CC3D package.

## Windows

On Windows, the script uses MSVC. You do not need to open a Visual Studio Developer Command Prompt first. If `cl.exe` is not already available, the script locates Visual Studio with `vswhere.exe`, activates `vcvars64.bat` internally, and runs the compiler in that environment.

List available Visual Studio installations:

```bat
python -m cc3d.scripts.compile_steppable --list-msvc
```

Build with the latest detected Visual Studio:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp ^
  --repo C:\path\to\compucell3d_kernel
```

Build with a specific Visual Studio major version:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp ^
  --repo C:\path\to\compucell3d_kernel ^
  --msvc-version 18
```

Common versions:

- `--msvc-version 16`: Visual Studio 2019
- `--msvc-version 17`: Visual Studio 2022
- `--msvc-version 18`: Visual Studio 2026

For best ABI compatibility, use the same MSVC version that was used to build the installed CC3D package. If CC3D was built with Visual Studio 2019, prefer `--msvc-version 16`.

Example using the bundled demo:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\Users\m\src\compucell3d_kernel\CompuCell3D\core\Demos\SteppablesDemosCompiled\CellFieldOperations\Native\CellFieldOperationsSteppableCompiled.cpp ^
  --repo C:\Users\m\src\compucell3d_kernel ^
  --msvc-version 16 ^
  --verbose
```

If you already have a compiler configured, you can set `CXX` explicitly:

```bat
set CXX=C:\path\to\cl.exe
python -m cc3d.scripts.compile_steppable C:\path\to\YourSteppable.cpp --repo C:\path\to\compucell3d_kernel
```

## macOS

By default, macOS builds use Apple `clang++` and the active macOS SDK discovered through `xcrun`.

```bash
python -m cc3d.scripts.compile_steppable \
  /path/to/YourSteppable.cpp \
  --repo /path/to/compucell3d_kernel
```

The default macOS mode requires Xcode Command Line Tools:

```bash
xcode-select --install
```

You can also request the conda compiler toolchain:

```bash
python -m cc3d.scripts.compile_steppable \
  /path/to/YourSteppable.cpp \
  --repo /path/to/compucell3d_kernel \
  --macos-toolchain conda
```

If needed, set `CXX` to a specific compiler:

```bash
export CXX=/path/to/clang++
python -m cc3d.scripts.compile_steppable /path/to/YourSteppable.cpp --repo /path/to/compucell3d_kernel
```

## Cleaning

Remove the compiled output next to the source file:

```bash
python -m cc3d.scripts.compile_steppable /path/to/YourSteppable.cpp --repo /path/to/compucell3d_kernel --clean
```

On Windows, use the same command form with Windows paths.

## Notes

The compiled source must export the CC3D entry point:

```cpp
extern "C" CC3D_STEPPABLE_EXPORT const CC3DSteppableV1 *cc3d_get_steppable_v1() {
    return &api;
}
```

`CC3D_STEPPABLE_EXPORT` is required on Windows so the function appears in the DLL export table and can be found by `GetProcAddress`.

## Tool Integration

Player and Twedit++ can query Visual Studio compiler options without parsing command-line output:

```python
from cc3d.scripts.windows_vcvars import msvc_compiler_info

compilers = [info.to_dict() for info in msvc_compiler_info(probe=True)]
```

Each result contains:

- `available`: whether `vcvars64.bat` could be activated and `cl.exe` located
- `selected`: whether this install matches the requested version
- `installation_path`
- `installation_version`
- `display_name`
- `vcvars64`
- `compiler_path`
- `compiler_version`
- `error`

To select a specific Visual Studio major version:

```python
compilers = [info.to_dict() for info in msvc_compiler_info("18", probe=True)]
```

For GUI use, call with `probe=False` to populate a fast initial selector, then call with `probe=True` when the user selects a compiler or starts a build.
