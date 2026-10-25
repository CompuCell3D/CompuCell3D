from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METADATA_SCHEMA_VERSION = 2
METADATA_SUFFIX = ".cc3d-build.json"


def build_artifacts_dir(output_path: Path | str) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    return output_path.parent / ".cc3d-build" / output_path.stem


def metadata_path(output_path: Path | str) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    return build_artifacts_dir(output_path) / f"{output_path.stem}{METADATA_SUFFIX}"


def file_sha256(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_abi_version(kernel_header: Path | str) -> int | None:
    path = Path(kernel_header)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"#define\s+CC3D_KERNEL_ABI_VERSION\s+(\d+)u", text)
    return int(match.group(1)) if match else None


def write_build_metadata(
    source_path: Path | str,
    output_path: Path | str,
    include_root: Path | str,
    include_root_source: str,
    command: tuple[str, ...] | list[str],
    build_type: str,
    source_root: Path | str | None = None,
    source_paths: list[Path | str] | tuple[Path | str, ...] | None = None,
    header_paths: list[Path | str] | tuple[Path | str, ...] | None = None,
    include_dirs: list[Path | str] | tuple[Path | str, ...] | None = None,
) -> Path:
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    include_root = Path(include_root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve() if source_root else source_path.parent
    source_paths = tuple(Path(path).expanduser().resolve() for path in (source_paths or (source_path,)))
    header_paths = tuple(Path(path).expanduser().resolve() for path in (header_paths or ()))
    include_dirs = tuple(Path(path).expanduser().resolve() for path in (include_dirs or (include_root,)))
    kernel_header = include_root / "cc3d" / "kernel.h"
    metadata = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_root": str(source_root),
        "source_paths": [str(path) for path in source_paths],
        "header_paths": [str(path) for path in header_paths],
        "source_hashes": {str(path): file_sha256(path) for path in source_paths},
        "header_hashes": {str(path): file_sha256(path) for path in header_paths},
        "include_dirs": [str(path) for path in include_dirs],
        "output_path": str(output_path),
        "metadata_path": str(metadata_path(output_path)),
        "source_sha256": file_sha256(source_path),
        "output_sha256": file_sha256(output_path),
        "kernel_header": str(kernel_header),
        "kernel_header_sha256": file_sha256(kernel_header),
        "kernel_abi_version": kernel_abi_version(kernel_header),
        "include_root": str(include_root),
        "include_root_source": include_root_source,
        "build_type": build_type,
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler_command": list(command),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = metadata_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)
    return path


def read_build_metadata(output_path: Path | str) -> dict[str, Any] | None:
    path = metadata_path(output_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None
