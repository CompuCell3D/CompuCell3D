from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cc3d.core.DefaultSettingsData import SETTINGS_FOLDER


SETTINGS_FILE_NAME = "compiled_steppables.yaml"


def settings_path() -> Path:
    return Path.home() / SETTINGS_FOLDER / SETTINGS_FILE_NAME


def _default_settings() -> dict[str, Any]:
    return {
        "version": 1,
        "auto_compile": {
            "enabled": True,
            "compile_if": {
                "output_missing": True,
                "source_newer": True,
                "kernel_header_newer": True,
            },
        },
        "compiler": {
            "selection_policy": "latest",
        },
        "platforms": {
            "windows": {
                "family": "msvc",
                "selection_policy": "latest",
                "preferred": {
                    "visual_studio_major": None,
                    "installation_path": None,
                    "display_name": None,
                    "vcvars64": None,
                    "compiler_path": None,
                },
                "detected": None,
            },
            "macos": {
                "family": "clang",
                "selection_policy": "apple",
                "preferred": {
                    "toolchain": "apple",
                    "compiler_path": None,
                },
                "detected": None,
            },
            "linux": {
                "family": "cxx",
                "selection_policy": "latest",
                "preferred": {
                    "compiler_path": None,
                },
                "detected": None,
            },
        },
    }


def load_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or settings_path()
    if not path.exists():
        settings = _default_settings()
        refresh_detected_compiler(settings)
        save_settings(settings, path)
        return settings

    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp) or {}

    settings = _default_settings()
    _deep_update(settings, loaded)
    return settings


def save_settings(settings: dict[str, Any], path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(settings, fp, sort_keys=False)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def current_platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def refresh_detected_compiler(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or _default_settings()
    platform_key = current_platform_key()
    platform_settings = settings["platforms"][platform_key]
    platform_settings["detected"] = _detect_current_platform_compiler(platform_key)
    return settings


def auto_compile_enabled(path: Path | None = None) -> bool:
    settings = load_settings(path)
    return bool(settings.get("auto_compile", {}).get("enabled", True))


def _detect_current_platform_compiler(platform_key: str) -> dict[str, Any] | None:
    if platform_key == "windows":
        from cc3d.scripts.windows_vcvars import msvc_compiler_info

        infos = [info for info in msvc_compiler_info(probe=True) if info.available]
        if not infos:
            return None
        selected = next((info for info in infos if info.selected), infos[0])
        major = selected.installation_version.split(".", 1)[0] if selected.installation_version else None
        return {
            "visual_studio_major": major,
            "installation_version": selected.installation_version,
            "display_name": selected.display_name,
            "installation_path": selected.installation_path,
            "vcvars64": selected.vcvars64,
            "compiler_path": selected.compiler_path,
            "compiler_version": selected.compiler_version,
        }

    compiler_path = shutil.which("clang++" if platform_key == "macos" else "g++") or shutil.which("clang++")
    if not compiler_path:
        return None
    return {
        "compiler_path": compiler_path,
    }


@dataclass(frozen=True)
class CompilerSelection:
    msvc_version: str | None = None
    macos_toolchain: str | None = None
    warnings: tuple[str, ...] = ()
    settings_file: Path | None = None


def compiler_selection_from_settings(
    explicit_msvc_version: str | None = None,
    explicit_macos_toolchain: str | None = None,
    path: Path | None = None,
) -> CompilerSelection:
    warnings: list[str] = []
    if explicit_msvc_version or explicit_macos_toolchain:
        return CompilerSelection(
            msvc_version=explicit_msvc_version,
            macos_toolchain=explicit_macos_toolchain,
            warnings=tuple(warnings),
            settings_file=path or settings_path(),
        )

    path = path or settings_path()
    settings = load_settings(path)
    platform_key = current_platform_key()
    platform_settings = settings["platforms"][platform_key]

    if platform_key == "windows":
        return _windows_selection_from_settings(settings, path)
    if platform_key == "macos":
        toolchain = platform_settings.get("selection_policy") or platform_settings.get("preferred", {}).get("toolchain") or "apple"
        if toolchain not in ("apple", "conda"):
            warnings.append(
                f"WARNING: Invalid macOS compiled-steppable toolchain '{toolchain}' in {path}; using 'apple'."
            )
            toolchain = "apple"
        refresh_detected_compiler(settings)
        save_settings(settings, path)
        return CompilerSelection(macos_toolchain=toolchain, warnings=tuple(warnings), settings_file=path)

    refresh_detected_compiler(settings)
    save_settings(settings, path)
    return CompilerSelection(warnings=tuple(warnings), settings_file=path)


def _windows_selection_from_settings(settings: dict[str, Any], path: Path) -> CompilerSelection:
    from cc3d.scripts.windows_vcvars import msvc_compiler_info

    warnings: list[str] = []
    platform_settings = settings["platforms"]["windows"]
    policy = platform_settings.get("selection_policy") or settings.get("compiler", {}).get("selection_policy") or "latest"
    preferred = platform_settings.get("preferred") or {}
    msvc_version: str | None = None

    if policy == "visual_studio_major":
        preferred_major = preferred.get("visual_studio_major")
        if preferred_major:
            infos = msvc_compiler_info(msvc_version=str(preferred_major), probe=True)
            if any(info.available and info.selected for info in infos):
                msvc_version = str(preferred_major)
            else:
                warnings.append(
                    "WARNING: Preferred Visual Studio compiler "
                    f"'{preferred_major}' from {path} is not available; using latest detected compiler."
                )
                platform_settings["selection_policy"] = "latest"

    elif policy != "latest":
        warnings.append(
            f"WARNING: Invalid Windows compiled-steppable compiler policy '{policy}' in {path}; "
            "using latest detected compiler."
        )
        platform_settings["selection_policy"] = "latest"

    refresh_detected_compiler(settings)
    save_settings(settings, path)
    return CompilerSelection(msvc_version=msvc_version, warnings=tuple(warnings), settings_file=path)
