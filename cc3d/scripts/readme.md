# Compiling Native Steppables

`compile_steppable.py` builds a C++ steppable that uses the public `cc3d/kernel.h` ABI. The source file passed on the command line is the entry source. Other `.cpp`, `.h`, and `.hpp` files below its directory are discovered automatically and compiled or tracked as part of the same native target. The output is written next to the entry source file:

- Windows: `.dll`
- macOS: `.dylib`
- Linux: `.so`

For example, this source tree produces one shared library:

```text
Native_GrowthSteppable/
    GrowthSteppable.cpp
    Utilities/
        Vector3.cpp
        Geometry.h
    Models/
        CellGrowthModel.cpp
        CellGrowthModel.h
```

The `.cc3d` resource declaration remains a reference to the entry source:

```xml
<Resource Type="Cpp">Native_GrowthSteppable/GrowthSteppable.cpp</Resource>
```

The directory containing the entry source is the target root. Include paths for the target root and discovered subdirectories are added automatically. Keep separate compiled steppables in separate target roots; two C++ resources sharing one target root are rejected.

Run the command from an activated CC3D conda environment so `python -m cc3d.scripts.compile_steppable` resolves to the installed CC3D package. In an installed conda package, the script expects the public ABI header at `$CONDA_PREFIX/include/cc3d/kernel.h`.

## Windows

On Windows, the script uses MSVC. You do not need to open a Visual Studio Developer Command Prompt first. If `cl.exe` is not already available, the script locates Visual Studio with `vswhere.exe`, activates `vcvars64.bat` internally, and runs the compiler in that environment.

List available Visual Studio installations:

```bat
python -m cc3d.scripts.compile_steppable --list-msvc
```

Build with the latest detected Visual Studio:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp
```

Build with a specific Visual Studio major version:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp ^
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
  --msvc-version 16 ^
  --verbose
```

If you already have a compiler configured, you can set `CXX` explicitly:

```bat
set CXX=C:\path\to\cl.exe
python -m cc3d.scripts.compile_steppable C:\path\to\YourSteppable.cpp
```

For source-tree development before headers are installed, pass the repository path:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp ^
  --repo C:\path\to\compucell3d_kernel
```

You can also pass an explicit include root containing `cc3d/kernel.h`:

```bat
python -m cc3d.scripts.compile_steppable ^
  C:\path\to\YourSteppable.cpp ^
  --include-root C:\path\to\include
```

## macOS

By default, macOS builds use Apple `clang++` and the active macOS SDK discovered through `xcrun`.

```bash
python -m cc3d.scripts.compile_steppable \
  /path/to/YourSteppable.cpp
```

The default macOS mode requires Xcode Command Line Tools:

```bash
xcode-select --install
```

You can also request the conda compiler toolchain:

```bash
python -m cc3d.scripts.compile_steppable \
  /path/to/YourSteppable.cpp \
  --macos-toolchain conda
```

If needed, set `CXX` to a specific compiler:

```bash
export CXX=/path/to/clang++
python -m cc3d.scripts.compile_steppable /path/to/YourSteppable.cpp
```

For source-tree development before headers are installed, pass `--repo /path/to/compucell3d_kernel` or `--include-root /path/to/include`.

## Cleaning

Remove the compiled output next to the source file:

```bash
python -m cc3d.scripts.compile_steppable /path/to/YourSteppable.cpp --clean
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

## Automatic Compilation for `.cc3d` Projects

`cc3d.run_script` automatically compiles C++ steppable resources listed in a `.cc3d` file before starting the simulation. It looks for resources such as:

```xml
<Resource Type="Cpp">Native/CellFieldOperationsSteppableCompiled.cpp</Resource>
```

Resource paths are resolved relative to the folder containing the `.cc3d` project file.

Automatic compilation only rebuilds when needed:

- the native library does not exist
- the declared entry source, a discovered implementation source, or a discovered header is newer than or differs from the recorded build metadata
- the discovered source set or include paths changed
- `cc3d/kernel.h` is newer than the native library
- `--force-compile-steppables` is passed

Run a project with automatic compilation:

```bash
python -m cc3d.run_script -i /path/to/project.cc3d
```

Disable automatic compilation:

```bash
python -m cc3d.run_script -i /path/to/project.cc3d --no-compile-steppables
```

Force compilation:

```bash
python -m cc3d.run_script -i /path/to/project.cc3d --force-compile-steppables
```

Windows compiler selection:

```bat
python -m cc3d.run_script ^
  -i C:\path\to\project.cc3d ^
  --steppable-msvc-version 16
```

Client code can use the same project-level API:

```python
from cc3d.core.CompiledSteppableAutoCompiler import (
    compiled_steppable_build_plan,
    compile_project_steppables,
    format_compiled_steppable_summary,
)

plans = compiled_steppable_build_plan("/path/to/project.cc3d")
summary = compile_project_steppables("/path/to/project.cc3d", msvc_version="16")
print(format_compiled_steppable_summary(summary))
```

The summary object exposes `to_dict()` for GUI clients that need structured status instead of formatted terminal output.

## Compiler Settings File

The first automatic compilation run creates a user settings file:

```text
~/.compucell3d_py3/compiled_steppables.yaml
```

On Windows this is typically:

```text
C:\Users\<user>\.compucell3d_py3\compiled_steppables.yaml
```

The default policy is `latest`, meaning CC3D uses the latest compiler it can detect on the current platform. The generated file also stores a `detected` section for diagnostics.

Example Windows configuration:

```yaml
version: 1
auto_compile:
  enabled: true
  compile_if:
    output_missing: true
    source_newer: true
    kernel_header_newer: true
compiler:
  selection_policy: latest
platforms:
  windows:
    family: msvc
    selection_policy: latest
    preferred:
      visual_studio_major: null
      installation_path: null
      display_name: null
      vcvars64: null
      compiler_path: null
    detected:
      visual_studio_major: '18'
      installation_version: 18.9.12120.119
      display_name: Visual Studio Community 2026
      installation_path: C:\Program Files\Microsoft Visual Studio\18\Community
      vcvars64: C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
      compiler_path: C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\...\cl.exe
      compiler_version: Microsoft (R) C/C++ Optimizing Compiler Version ... for x64
```

To pin Windows builds to a Visual Studio major version, use:

```yaml
platforms:
  windows:
    selection_policy: visual_studio_major
    preferred:
      visual_studio_major: '16'
```

CC3D validates stored compiler preferences on each automatic build. If the stored compiler is unavailable, it prints a warning and falls back to the latest detected compiler.

Command-line options have priority over this file. For example, `--steppable-msvc-version 16` uses Visual Studio 2019 for that run regardless of the YAML setting.
