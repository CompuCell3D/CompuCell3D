from __future__ import annotations

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisualStudioInstall:
    installation_path: Path
    installation_version: str
    display_name: str

    @property
    def vcvars64(self) -> Path:
        return self.installation_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"


@dataclass(frozen=True)
class MsvcCompilerInfo:
    available: bool
    selected: bool
    installation_path: str
    installation_version: str
    display_name: str
    vcvars64: str
    compiler_path: str | None = None
    compiler_version: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "available": self.available,
            "selected": self.selected,
            "installation_path": self.installation_path,
            "installation_version": self.installation_version,
            "display_name": self.display_name,
            "vcvars64": self.vcvars64,
            "compiler_path": self.compiler_path,
            "compiler_version": self.compiler_version,
            "error": self.error,
        }


def _vswhere_path() -> Path | None:
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    candidates = []
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    candidates.append(Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_vswhere(args: list[str]) -> str:
    vswhere = _vswhere_path()
    if not vswhere:
        raise RuntimeError("Could not locate vswhere.exe from Visual Studio Installer.")
    return subprocess.check_output([str(vswhere), *args], text=True, stderr=subprocess.STDOUT)


def env_value(env: dict[str, str], name: str) -> str | None:
    for key, value in env.items():
        if key.upper() == name.upper():
            return value
    return None


def list_visual_studio_installs() -> list[VisualStudioInstall]:
    output = _run_vswhere([
        "-all",
        "-prerelease",
        "-products",
        "*",
        "-format",
        "json",
    ])

    installs = []
    for item in json.loads(output):
        install = VisualStudioInstall(
            Path(item["installationPath"]),
            item["installationVersion"],
            item.get("displayName", "Visual Studio"),
        )
        installs.append(install)
    installs_by_path = {install.installation_path: install for install in installs}
    for root in (Path("C:/Program Files/Microsoft Visual Studio"), Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if not root.exists():
            continue
        for vcvars64 in root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"):
            installation_path = vcvars64.parents[3]
            if installation_path in installs_by_path:
                continue
            version = installation_path.parent.name
            display_name = f"Visual Studio {version} {installation_path.name}"
            installs_by_path[installation_path] = VisualStudioInstall(installation_path, version, display_name)
    installs = list(installs_by_path.values())
    return installs


def select_visual_studio_install(msvc_version: str | None = None) -> VisualStudioInstall:
    installs = list_visual_studio_installs()
    if not installs:
        raise RuntimeError("Could not locate a Visual Studio installation with the x64 C++ tools.")

    if not msvc_version:
        return sorted(installs, key=lambda item: item.installation_version, reverse=True)[0]

    matches = [
        install for install in installs
        if install.installation_version.startswith(msvc_version)
        or f"Visual Studio\\{msvc_version}\\" in str(install.installation_path)
        or f"\\{msvc_version}\\" in str(install.installation_path)
    ]
    if matches:
        return sorted(matches, key=lambda item: item.installation_version, reverse=True)[0]

    available = "\n".join(
        f"  {install.installation_version}: {install.display_name} at {install.installation_path}"
        for install in installs
    )
    raise RuntimeError(f"Could not find Visual Studio version '{msvc_version}'. Available installations:\n{available}")


def environment_from_vcvars64(vcvars64: Path) -> dict[str, str]:
    batch_text = f'@echo off\ncall "{vcvars64}" >nul\nif errorlevel 1 exit /b %errorlevel%\nset\n'
    batch_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False) as batch_file:
            batch_file.write(batch_text)
            batch_path = batch_file.name
        output = subprocess.check_output(["cmd.exe", "/d", "/c", batch_path], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to activate Visual Studio environment with {vcvars64}:\n{e.output}") from e
    finally:
        if batch_path:
            Path(batch_path).unlink(missing_ok=True)
    env = os.environ.copy()
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value
            if key.upper() == "PATH":
                env["PATH"] = value
                env["Path"] = value
    return env


def compiler_path_from_env(env: dict[str, str]) -> str | None:
    vc_tools = env_value(env, "VCToolsInstallDir")
    if vc_tools:
        for host_dir in ("HostX64", "Hostx64"):
            vc_cl = Path(vc_tools) / "bin" / host_dir / "x64" / "cl.exe"
            if vc_cl.exists():
                return str(vc_cl)

    path_value = env_value(env, "PATH") or ""
    for path_item in path_value.split(os.pathsep):
        if path_item.lower().endswith("\\bin\\hostx64\\x64"):
            return str(Path(path_item) / "cl.exe")

    return None


def compiler_version_from_env(env: dict[str, str], compiler_path: str | None = None) -> str | None:
    compiler = "cl.exe" if compiler_path else (compiler_path_from_env(env) or "cl.exe")
    compiler_command = compiler if compiler.lower() == "cl.exe" else f'"{compiler}"'
    try:
        output = subprocess.check_output(
            ["cmd.exe", "/d", "/c", compiler_command],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        output = getattr(e, "output", "")

    for line in output.splitlines():
        if "C/C++" in line and "Version" in line:
            return line.strip()
    return None


def msvc_compiler_info(msvc_version: str | None = None, probe: bool = False) -> list[MsvcCompilerInfo]:
    try:
        installs = list_visual_studio_installs()
    except Exception as e:
        return [
            MsvcCompilerInfo(
                available=False,
                selected=False,
                installation_path="",
                installation_version="",
                display_name="Visual Studio",
                vcvars64="",
                error=str(e),
            )
        ]

    selected_install = None
    selection_error = None
    try:
        selected_install = select_visual_studio_install(msvc_version)
    except Exception as e:
        selection_error = str(e)

    results = []
    for install in sorted(installs, key=lambda item: item.installation_version, reverse=True):
        selected = selected_install is not None and install.installation_path == selected_install.installation_path
        compiler_path = None
        compiler_version = None
        error = None
        available = True

        if probe or selected:
            try:
                env = environment_from_vcvars64(install.vcvars64)
                compiler_path = compiler_path_from_env(env)
                compiler_version = compiler_version_from_env(env, compiler_path)
                if not compiler_path:
                    available = False
                    error = "Could not locate cl.exe after activating vcvars64.bat."
            except Exception as e:
                available = False
                error = str(e)

        results.append(
            MsvcCompilerInfo(
                available=available,
                selected=selected,
                installation_path=str(install.installation_path),
                installation_version=install.installation_version,
                display_name=install.display_name,
                vcvars64=str(install.vcvars64),
                compiler_path=compiler_path,
                compiler_version=compiler_version,
                error=error,
            )
        )

    if selection_error and not results:
        return [
            MsvcCompilerInfo(
                available=False,
                selected=False,
                installation_path="",
                installation_version="",
                display_name="Visual Studio",
                vcvars64="",
                error=selection_error,
            )
        ]
    return results
