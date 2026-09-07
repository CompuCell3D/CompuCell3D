#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from cc3d.scripts.windows_vcvars import (
        compiler_path_from_env,
        environment_from_vcvars64,
        msvc_compiler_info,
        select_visual_studio_install,
    )
except ImportError:
    from windows_vcvars import (
        compiler_path_from_env,
        environment_from_vcvars64,
        msvc_compiler_info,
        select_visual_studio_install,
    )


DEFAULT_REPO = None


@dataclass(frozen=True)
class IncludeRootInfo:
    include_root: Path
    kernel_header: Path
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "include_root": str(self.include_root),
            "kernel_header": str(self.kernel_header),
            "source": self.source,
        }


@dataclass(frozen=True)
class CompileResult:
    source_path: Path
    output_path: Path
    include_root: Path
    include_root_source: str
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.return_code == 0

    @property
    def log(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "include_root": str(self.include_root),
            "include_root_source": self.include_root_source,
            "command": list(self.command),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "success": self.success,
        }


def env_value(env: dict[str, str], name: str) -> str | None:
    for key, value in env.items():
        if key.upper() == name.upper():
            return value
    return None


def platform_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def normalized_build_type(build_type: str) -> str:
    value = build_type.lower().replace("_", "")
    if value not in {"debug", "release", "relwithdebinfo"}:
        raise ValueError("build_type must be Debug, Release, or RelWithDebInfo")
    return value


def compiler_debug_flags(build_type: str, compiler: str) -> list[str]:
    value = normalized_build_type(build_type)
    if compiler == "msvc":
        return ["/Od", "/Zi"] if value == "debug" else (["/O2", "/Zi"] if value == "relwithdebinfo" else ["/O2"])
    return ["-O0", "-g"] if value == "debug" else (["-O2", "-g"] if value == "relwithdebinfo" else ["-O3"])


def resolve_source(repo_path: Path, source_arg: str) -> Path:
    candidate = Path(source_arg).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (repo_path / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    matches = sorted(repo_path.rglob(candidate.name))
    matches = [match for match in matches if match.is_file()]

    if not matches:
        raise FileNotFoundError(f"Could not locate steppable source '{source_arg}' under {repo_path}")
    if len(matches) > 1:
        match_list = "\n".join(str(match) for match in matches)
        raise RuntimeError(
            f"Source name '{source_arg}' is ambiguous under {repo_path}. "
            f"Pass a more specific path.\nMatches:\n{match_list}"
        )

    return matches[0]


def _kernel_header_at(include_root: Path) -> Path | None:
    kernel_header = include_root / "cc3d" / "kernel.h"
    if kernel_header.exists():
        return kernel_header
    return None


def resolve_kernel_include_root(
    repo_path: Path | None = None,
    include_root: Path | None = None,
) -> IncludeRootInfo:
    checked = []

    if include_root:
        candidate = include_root.expanduser().resolve()
        checked.append(candidate / "cc3d" / "kernel.h")
        kernel_header = _kernel_header_at(candidate)
        if kernel_header:
            return IncludeRootInfo(candidate, kernel_header, "explicit")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "include"
        checked.append(candidate / "cc3d" / "kernel.h")
        kernel_header = _kernel_header_at(candidate)
        if kernel_header:
            return IncludeRootInfo(candidate, kernel_header, "conda-prefix")

    candidate = Path(sys.prefix) / "include"
    checked.append(candidate / "cc3d" / "kernel.h")
    kernel_header = _kernel_header_at(candidate)
    if kernel_header:
        return IncludeRootInfo(candidate, kernel_header, "sys-prefix")

    if repo_path:
        candidate = repo_path.expanduser().resolve() / "CompuCell3D" / "core"
        checked.append(candidate / "cc3d" / "kernel.h")
        kernel_header = _kernel_header_at(candidate)
        if kernel_header:
            return IncludeRootInfo(candidate, kernel_header, "repo")

    checked_paths = "\n".join(f"  {path}" for path in checked)
    raise FileNotFoundError(
        "Could not locate cc3d/kernel.h. Checked:\n"
        f"{checked_paths}\n\n"
        "Install CC3D headers into the active environment, pass --include-root, "
        "or pass --repo for a source checkout."
    )


def build_output_path(source_path: Path) -> Path:
    return source_path.with_suffix(platform_suffix())


def conda_compiler() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        compiler = Path(conda_prefix) / "bin" / "clang++"
        if compiler.exists():
            return str(compiler)

    compiler = os.environ.get("CXX")
    if compiler:
        return compiler

    return shutil.which("clang++")


def macos_compile_command(
    source_path: Path,
    include_root: Path,
    output_path: Path,
    build_type: str = "Release",
) -> list[str]:

    try:
        compiler = subprocess.check_output(
            ["xcrun", "--find", "clang++"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

        sdkroot = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Apple Xcode Command Line Tools are required to compile "
            "CC3D native steppables.\n\n"
            "Install them with:\n"
            "    xcode-select --install"
        ) from e

    if not compiler:
        raise RuntimeError(
            "Could not locate Apple clang++. "
            "Install Xcode Command Line Tools with: xcode-select --install"
        )

    if not sdkroot:
        raise RuntimeError(
            "Could not locate the macOS SDK. "
            "Install Xcode Command Line Tools with: xcode-select --install"
        )

    libcxx_include = Path(sdkroot) / "usr" / "include" / "c++" / "v1"

    if not libcxx_include.exists():
        raise RuntimeError(
            f"Could not locate libc++ headers at:\n"
            f"    {libcxx_include}\n\n"
            "Your Xcode Command Line Tools installation may be incomplete."
        )

    return [
        compiler,
        *compiler_debug_flags(build_type, "clang"),
        "-std=c++17",
        "-stdlib=libc++",
        "-dynamiclib",

        "-isysroot",
        sdkroot,

        "-isystem",
        str(libcxx_include),

        f"-I{include_root}",

        str(source_path),

        "-o",
        str(output_path),
    ]



def macos_compile_command_conda(source_path: Path, include_root: Path, output_path: Path, build_type: str = "Release") -> list[str]:
    compiler = conda_compiler()
    if not compiler:
        raise RuntimeError("Could not locate a conda clang++. Activate the conda environment or set CXX.")

    return [
        compiler,
        *compiler_debug_flags(build_type, "clang"),
        "-std=c++17",
        "-dynamiclib",
        str(source_path),
        f"-I{include_root}",
        "-o",
        str(output_path),
    ]


def linux_compile_command(source_path: Path, include_root: Path, output_path: Path, build_type: str = "Release") -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("Could not locate a C++ compiler. Set CXX or install g++/clang++.")

    return [
        compiler,
        *compiler_debug_flags(build_type, "clang"),
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(source_path),
        f"-I{include_root}",
        "-o",
        str(output_path),
    ]


def windows_compile_command(
    source_path: Path,
    include_root: Path,
    output_path: Path,
    msvc_version: str | None,
    build_type: str = "Release",
) -> tuple[list[str], dict[str, str] | None]:
    compiler = os.environ.get("CXX") or shutil.which("cl")
    env = None
    if not compiler:
        install = select_visual_studio_install(msvc_version)
        env = environment_from_vcvars64(install.vcvars64)
        compiler = compiler_path_from_env(env)
        if not compiler:
            raise RuntimeError(f"Could not locate cl.exe after activating {install.vcvars64}")

    return [
        compiler,
        *compiler_debug_flags(build_type, "msvc"),
        "/std:c++17",
        "/EHsc",
        "/LD",
        *(["/DEBUG"] if normalized_build_type(build_type) in {"debug", "relwithdebinfo"} else []),
        str(source_path),
        f"/I{include_root}",
        f"/Fo:{output_path.with_suffix('.obj')}",
        f"/Fe:{output_path}",
    ], env


def compile_command(
    source_path: Path,
    include_root: Path,
    output_path: Path,
    macos_toolchain: str,
    msvc_version: str | None,
    build_type: str = "Release",
) -> tuple[list[str], dict[str, str] | None]:
    if sys.platform == "darwin":
        if macos_toolchain == "conda":
            return macos_compile_command_conda(
                source_path=source_path,
                include_root=include_root,
                output_path=output_path,
                build_type=build_type,
            ), None
        return macos_compile_command(source_path=source_path, include_root=include_root, output_path=output_path, build_type=build_type), None
    if sys.platform.startswith("win"):
        return windows_compile_command(
            source_path=source_path,
            include_root=include_root,
            output_path=output_path,
            msvc_version=msvc_version,
            build_type=build_type,
        )
    return linux_compile_command(source_path=source_path, include_root=include_root, output_path=output_path, build_type=build_type), None


def clean_output(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"Removed {output_path}")
    else:
        print(f"Nothing to clean: {output_path}")


def _effective_compile_command(cmd: list[str], env: dict[str, str] | None) -> list[str]:
    if sys.platform.startswith("win") and env is not None:
        cmd = ["cl.exe", *cmd[1:]]
        return ["cmd.exe", "/d", "/c", subprocess.list2cmdline(cmd)]
    return cmd


def run_compile_command(cmd: list[str], env: dict[str, str] | None) -> None:
    subprocess.run(_effective_compile_command(cmd, env), check=True, env=env)


def compile_steppable_source(
    source_path: Path | str,
    repo_path: Path | str | None = None,
    include_root: Path | str | None = None,
    output_path: Path | str | None = None,
    macos_toolchain: str = "apple",
    msvc_version: str | None = None,
    build_type: str = "Release",
) -> CompileResult:
    resolved_repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
    resolved_source_path = Path(source_path).expanduser().resolve()
    resolved_output_path = Path(output_path).expanduser().resolve() if output_path else build_output_path(resolved_source_path)
    include_root_info = resolve_kernel_include_root(
        repo_path=resolved_repo_path,
        include_root=Path(include_root) if include_root else None,
    )

    cmd, env = compile_command(
        source_path=resolved_source_path,
        include_root=include_root_info.include_root,
        output_path=resolved_output_path,
        macos_toolchain=macos_toolchain,
        msvc_version=msvc_version,
        build_type=build_type,
    )
    effective_cmd = _effective_compile_command(cmd, env)
    completed = subprocess.run(
        effective_cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CompileResult(
        source_path=resolved_source_path,
        output_path=resolved_output_path,
        include_root=include_root_info.include_root,
        include_root_source=include_root_info.source,
        command=tuple(effective_cmd),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a standalone CompuCell3D compiled steppable against cc3d/kernel.h"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Steppable C++ source file. May be an absolute path, relative path, or basename such as GrowthSteppable.cpp."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Path to the CompuCell3D repository. Used as a fallback for source-tree development."
    )
    parser.add_argument(
        "--include-root",
        default=None,
        help="Include root containing cc3d/kernel.h. Overrides conda/sys prefix and --repo."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the compiled extension instead of building it."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full compiler command before running it."
    )
    parser.add_argument(
        "--build-type",
        choices=("Debug", "Release", "RelWithDebInfo"),
        default="Release",
        help="Compiler configuration. RelWithDebInfo enables optimization and debug symbols. Default: Release."
    )
    parser.add_argument(
        "--macos-toolchain",
        choices=("apple", "conda"),
        default="apple",
        help="macOS only: choose the Apple or conda compiler toolchain. Default: apple"
    )
    parser.add_argument(
        "--msvc-version",
        default=None,
        help="Windows only: Visual Studio major version to use, such as 18 or 17. Defaults to latest."
    )
    parser.add_argument(
        "--list-msvc",
        action="store_true",
        help="Windows only: list Visual Studio installations with x64 C++ tools and exit."
    )
    parser.add_argument(
        "--probe-msvc",
        action="store_true",
        help="Windows only: activate detected Visual Studio installations while listing compiler details."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_msvc:
        for info in msvc_compiler_info(msvc_version=args.msvc_version, probe=args.probe_msvc):
            marker = " [selected]" if info.selected else ""
            status = "available" if info.available else "unavailable"
            print(f"{info.installation_version}: {info.display_name}{marker} ({status})")
            print(f"  installation_path: {info.installation_path}")
            print(f"  vcvars64: {info.vcvars64}")
            if info.compiler_path:
                print(f"  compiler_path: {info.compiler_path}")
            if info.compiler_version:
                print(f"  compiler_version: {info.compiler_version}")
            if info.error:
                print(f"  error: {info.error}")
        return 0
    if not args.source:
        raise RuntimeError("Missing steppable source path.")

    repo_path = Path(args.repo).expanduser().resolve() if args.repo else None
    search_root = repo_path or Path.cwd()
    source_path = resolve_source(repo_path=search_root, source_arg=args.source)
    output_path = build_output_path(source_path)

    if args.clean:
        clean_output(output_path)
        return 0

    include_root_info = resolve_kernel_include_root(
        repo_path=repo_path,
        include_root=Path(args.include_root) if args.include_root else None,
    )

    cmd, env = compile_command(
        source_path=source_path,
        include_root=include_root_info.include_root,
        output_path=output_path,
        macos_toolchain=args.macos_toolchain,
        msvc_version=args.msvc_version,
        build_type=args.build_type,
    )

    print(f"Source : {source_path}")
    print(f"Output : {output_path}")
    if repo_path:
        print(f"Repo   : {repo_path}")
    print(f"Include: {include_root_info.include_root} ({include_root_info.source})")
    if args.verbose:
        print("Command:")
        print(" ".join(cmd))

    run_compile_command(cmd, env)
    print(f"Built {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
