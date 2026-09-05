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
