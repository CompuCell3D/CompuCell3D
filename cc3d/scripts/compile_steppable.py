#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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


DEFAULT_REPO = Path("/")


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
        "-O3",
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



def macos_compile_command_conda(source_path: Path, include_root: Path, output_path: Path) -> list[str]:
    compiler = conda_compiler()
    if not compiler:
        raise RuntimeError("Could not locate a conda clang++. Activate the conda environment or set CXX.")

    return [
        compiler,
        "-O3",
        "-std=c++17",
        "-dynamiclib",
        str(source_path),
        f"-I{include_root}",
        "-o",
        str(output_path),
    ]


def linux_compile_command(source_path: Path, include_root: Path, output_path: Path) -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("Could not locate a C++ compiler. Set CXX or install g++/clang++.")

    return [
        compiler,
        "-O3",
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
        "/O2",
        "/std:c++17",
        "/EHsc",
        "/LD",
        str(source_path),
        f"/I{include_root}",
        f"/Fe:{output_path}",
    ], env


def compile_command(
    source_path: Path,
    repo_path: Path,
    output_path: Path,
    macos_toolchain: str,
    msvc_version: str | None,
) -> tuple[list[str], dict[str, str] | None]:
    include_root = repo_path / "CompuCell3D" / "core"

    if sys.platform == "darwin":
        if macos_toolchain == "conda":
            return macos_compile_command_conda(
                source_path=source_path,
                include_root=include_root,
                output_path=output_path
            ), None
        return macos_compile_command(source_path=source_path, include_root=include_root, output_path=output_path), None
    if sys.platform.startswith("win"):
        return windows_compile_command(
            source_path=source_path,
            include_root=include_root,
            output_path=output_path,
            msvc_version=msvc_version,
        )
    return linux_compile_command(source_path=source_path, include_root=include_root, output_path=output_path), None


def clean_output(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"Removed {output_path}")
    else:
        print(f"Nothing to clean: {output_path}")


def run_compile_command(cmd: list[str], env: dict[str, str] | None) -> None:
    if sys.platform.startswith("win") and env is not None:
        cmd = ["cl.exe", *cmd[1:]]
        subprocess.run(["cmd.exe", "/d", "/c", subprocess.list2cmdline(cmd)], check=True, env=env)
        return
    subprocess.run(cmd, check=True, env=env)


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
        default=str(DEFAULT_REPO),
        help=f"Path to the CompuCell3D repository. Default: {DEFAULT_REPO}"
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

    repo_path = Path(args.repo).expanduser().resolve()
    source_path = resolve_source(repo_path=repo_path, source_arg=args.source)
    output_path = build_output_path(source_path)

    if args.clean:
        clean_output(output_path)
        return 0

    cmd, env = compile_command(
        source_path=source_path,
        repo_path=repo_path,
        output_path=output_path,
        macos_toolchain=args.macos_toolchain,
        msvc_version=args.msvc_version,
    )

    print(f"Source : {source_path}")
    print(f"Output : {output_path}")
    print(f"Repo   : {repo_path}")
    if args.verbose:
        print("Command:")
        print(" ".join(cmd))

    run_compile_command(cmd, env)
    print(f"Built {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
