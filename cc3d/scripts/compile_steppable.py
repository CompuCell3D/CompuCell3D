#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = Path("/")


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


def macos_compile_command(source_path: Path, include_root: Path, output_path: Path) -> list[str]:
    sdkroot = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        text=True
    ).strip()

    compiler = os.environ.get("CXX")
    if not compiler:
        compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()

    return [
        compiler,
        "-O3",
        "-std=c++17",
        "-stdlib=libc++",
        "-dynamiclib",
        str(source_path),
        f"-I{include_root}",
        "-isysroot",
        sdkroot,
        "-isystem",
        f"{sdkroot}/usr/include/c++/v1",
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


def windows_compile_command(source_path: Path, include_root: Path, output_path: Path) -> list[str]:
    compiler = os.environ.get("CXX") or shutil.which("cl")
    if not compiler:
        raise RuntimeError("Could not locate cl.exe. Run from a Visual Studio developer shell or set CXX.")

    return [
        compiler,
        "/O2",
        "/std:c++17",
        "/LD",
        str(source_path),
        f"/I{include_root}",
        f"/Fe:{output_path}",
    ]


def compile_command(source_path: Path, repo_path: Path, output_path: Path) -> list[str]:
    include_root = repo_path / "CompuCell3D" / "core"

    if sys.platform == "darwin":
        return macos_compile_command(source_path=source_path, include_root=include_root, output_path=output_path)
    if sys.platform.startswith("win"):
        return windows_compile_command(source_path=source_path, include_root=include_root, output_path=output_path)
    return linux_compile_command(source_path=source_path, include_root=include_root, output_path=output_path)


def clean_output(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"Removed {output_path}")
    else:
        print(f"Nothing to clean: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a standalone CompuCell3D compiled steppable against cc3d/kernel.h"
    )
    parser.add_argument(
        "source",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo).expanduser().resolve()
    source_path = resolve_source(repo_path=repo_path, source_arg=args.source)
    output_path = build_output_path(source_path)

    if args.clean:
        clean_output(output_path)
        return 0

    cmd = compile_command(source_path=source_path, repo_path=repo_path, output_path=output_path)

    print(f"Source : {source_path}")
    print(f"Output : {output_path}")
    print(f"Repo   : {repo_path}")
    if args.verbose:
        print("Command:")
        print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    print(f"Built {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
