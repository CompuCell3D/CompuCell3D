from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationDataHandler
from cc3d.core.CompiledSteppableCompilerSettings import compiler_selection_from_settings
from cc3d.core.CompiledSteppableMetadata import (
    file_sha256,
    kernel_abi_version,
    metadata_path,
    read_build_metadata,
)
from cc3d.scripts.compile_steppable import (
    CompileResult,
    build_output_path,
    clean_compiled_steppable_artifacts,
    compile_steppable_source,
    discover_steppable_sources,
    resolve_kernel_include_root,
    SteppableSourceSet,
)


@dataclass(frozen=True)
class CompiledSteppableBuildPlan:
    source_path: Path
    source_root: Path
    source_paths: tuple[Path, ...]
    header_paths: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    output_path: Path
    metadata_path: Path
    include_root: Path
    kernel_header: Path
    source_sha256: str | None
    kernel_header_sha256: str | None
    kernel_abi_version: int | None
    needs_compile: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "source_root": str(self.source_root),
            "source_paths": [str(path) for path in self.source_paths],
            "header_paths": [str(path) for path in self.header_paths],
            "include_dirs": [str(path) for path in self.include_dirs],
            "output_path": str(self.output_path),
            "metadata_path": str(self.metadata_path),
            "include_root": str(self.include_root),
            "kernel_header": str(self.kernel_header),
            "source_sha256": self.source_sha256,
            "kernel_header_sha256": self.kernel_header_sha256,
            "kernel_abi_version": self.kernel_abi_version,
            "needs_compile": self.needs_compile,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CompiledSteppableBuildSummary:
    cc3d_project_path: Path
    plans: tuple[CompiledSteppableBuildPlan, ...]
    results: tuple[CompileResult, ...]
    warnings: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return all(result.success for result in self.results)

    @property
    def compiled_count(self) -> int:
        return len(self.results)

    @property
    def skipped_count(self) -> int:
        return len([plan for plan in self.plans if not plan.needs_compile])

    def to_dict(self) -> dict[str, object]:
        return {
            "cc3d_project_path": str(self.cc3d_project_path),
            "plans": [plan.to_dict() for plan in self.plans],
            "results": [result.to_dict() for result in self.results],
            "warnings": list(self.warnings),
            "success": self.success,
            "compiled_count": self.compiled_count,
            "skipped_count": self.skipped_count,
        }


@dataclass(frozen=True)
class CompiledSteppableProjectStatus:
    """Metadata-backed status for every C++ steppable in a project."""

    cc3d_project_path: Path
    plans: tuple[CompiledSteppableBuildPlan, ...]

    @property
    def has_compiled_steppables(self) -> bool:
        return bool(self.plans)

    @property
    def needs_compile(self) -> bool:
        return any(plan.needs_compile for plan in self.plans)

    @property
    def valid(self) -> bool:
        return bool(self.plans) and not self.needs_compile

    def to_dict(self) -> dict[str, object]:
        return {
            "cc3d_project_path": str(self.cc3d_project_path),
            "plans": [plan.to_dict() for plan in self.plans],
            "has_compiled_steppables": self.has_compiled_steppables,
            "needs_compile": self.needs_compile,
            "valid": self.valid,
        }


def compiled_steppable_sources_from_cc3d(cc3d_project_path: Path | str) -> list[Path]:
    handler = CC3DSimulationDataHandler(None)
    handler.read_cc3_d_file_format(str(cc3d_project_path))
    resources = handler.cc3dSimulationData.getResourcesByType("Cpp")
    return [Path(resource.path).resolve() for resource in resources]


def compiled_steppable_build_plan(
    cc3d_project_path: Path | str,
    repo_path: Path | str | None = None,
    include_root: Path | str | None = None,
    force: bool = False,
) -> list[CompiledSteppableBuildPlan]:
    source_paths = compiled_steppable_sources_from_cc3d(cc3d_project_path)
    if not source_paths:
        return []

    include_root_info = resolve_kernel_include_root(
        repo_path=Path(repo_path) if repo_path else None,
        include_root=Path(include_root) if include_root else None,
    )
    kernel_header = include_root_info.kernel_header
    current_abi_version = kernel_abi_version(kernel_header)

    source_sets = [discover_steppable_sources(source_path) for source_path in source_paths]
    roots: dict[Path, Path] = {}
    for source_set in source_sets:
        overlapping_root = next(
            (
                root
                for root in roots
                if root == source_set.source_root
                or root in source_set.source_root.parents
                or source_set.source_root in root.parents
            ),
            None,
        )
        if overlapping_root is not None:
            previous = roots[overlapping_root]
            raise ValueError(
                "Multiple C++ resources have overlapping target roots "
                f"'{overlapping_root}' and '{source_set.source_root}': "
                f"'{previous}' and '{source_set.entry_source}'. "
                "Place each compiled steppable in its own source root."
            )
        roots[source_set.source_root] = source_set.entry_source

    plans: list[CompiledSteppableBuildPlan] = []
    for source_set in source_sets:
        source_path = source_set.entry_source
        output_path = build_output_path(source_path)
        build_metadata_path = metadata_path(output_path)
        source_hash = file_sha256(source_path)
        header_hash = file_sha256(kernel_header)
        source_hashes = {str(path): file_sha256(path) for path in source_set.source_paths}
        header_hashes = {str(path): file_sha256(path) for path in source_set.header_paths}
        common = {
            "source_path": source_path,
            "source_root": source_set.source_root,
            "source_paths": source_set.source_paths,
            "header_paths": source_set.header_paths,
            "include_dirs": source_set.include_dirs,
            "output_path": output_path,
            "metadata_path": build_metadata_path,
            "include_root": include_root_info.include_root,
            "kernel_header": kernel_header,
            "source_sha256": source_hash,
            "kernel_header_sha256": header_hash,
            "kernel_abi_version": current_abi_version,
        }
        if force:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="forced"))
            continue
        if not source_path.exists():
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="source missing"))
            continue
        if not output_path.exists():
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="output missing"))
            continue
        metadata = read_build_metadata(output_path)
        if metadata is None:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="metadata missing or invalid"))
            continue
        if metadata.get("source_sha256") != source_hash:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="source changed"))
            continue
        if metadata.get("source_root") != str(source_set.source_root):
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="source root changed"))
            continue
        if metadata.get("source_paths") != [str(path) for path in source_set.source_paths]:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="source set changed"))
            continue
        if metadata.get("header_paths") != [str(path) for path in source_set.header_paths]:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="header set changed"))
            continue
        if metadata.get("source_hashes") != source_hashes:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="source dependency changed"))
            continue
        if metadata.get("header_hashes") != header_hashes:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="header dependency changed"))
            continue
        if metadata.get("include_dirs") != [str(path) for path in source_set.include_dirs]:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="include paths changed"))
            continue
        if metadata.get("kernel_header_sha256") != header_hash:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="kernel header changed"))
            continue
        if metadata.get("kernel_abi_version") != current_abi_version:
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="kernel ABI changed"))
            continue
        if metadata.get("output_sha256") != file_sha256(output_path):
            plans.append(CompiledSteppableBuildPlan(**common, needs_compile=True, reason="output changed"))
            continue

        plans.append(CompiledSteppableBuildPlan(**common, needs_compile=False, reason="metadata matches"))

    return plans


def compiled_steppable_project_status(
    cc3d_project_path: Path | str,
    repo_path: Path | str | None = None,
    include_root: Path | str | None = None,
) -> CompiledSteppableProjectStatus:
    """Return current metadata-backed status without compiling or loading DLLs."""
    project_path = Path(cc3d_project_path).expanduser().resolve()
    plans = tuple(compiled_steppable_build_plan(
        cc3d_project_path=project_path,
        repo_path=repo_path,
        include_root=include_root,
        force=False,
    ))
    return CompiledSteppableProjectStatus(project_path, plans)


def clean_project_steppables(cc3d_project_path: Path | str) -> tuple[Path, ...]:
    """Remove generated artifacts for every C++ resource in a project."""
    project_path = Path(cc3d_project_path).expanduser().resolve()
    removed: list[Path] = []
    for source_path in compiled_steppable_sources_from_cc3d(project_path):
        removed.extend(clean_compiled_steppable_artifacts(build_output_path(source_path)))
    return tuple(removed)


def compile_project_steppables(
    cc3d_project_path: Path | str,
    repo_path: Path | str | None = None,
    include_root: Path | str | None = None,
    force: bool = False,
    macos_toolchain: str | None = None,
    msvc_version: str | None = None,
    compiler_settings_path: Path | str | None = None,
    build_type: str = "RelWithDebInfo",
) -> CompiledSteppableBuildSummary:
    project_path = Path(cc3d_project_path).expanduser().resolve()
    compiler_selection = compiler_selection_from_settings(
        explicit_msvc_version=msvc_version,
        explicit_macos_toolchain=macos_toolchain,
        path=Path(compiler_settings_path) if compiler_settings_path else None,
    )
    resolved_macos_toolchain = compiler_selection.macos_toolchain or "apple"
    resolved_msvc_version = compiler_selection.msvc_version

    plans = tuple(compiled_steppable_build_plan(
        cc3d_project_path=project_path,
        repo_path=repo_path,
        include_root=include_root,
        force=force,
    ))

    results: list[CompileResult] = []
    for plan in plans:
        if not plan.needs_compile:
            continue
        if not plan.source_path.exists():
            raise FileNotFoundError(f"Compiled steppable source does not exist: {plan.source_path}")
        result = compile_steppable_source(
            source_path=plan.source_path,
            repo_path=repo_path,
            include_root=include_root,
            output_path=plan.output_path,
            macos_toolchain=resolved_macos_toolchain,
            msvc_version=resolved_msvc_version,
            source_set=SteppableSourceSet(
                entry_source=plan.source_path,
                source_root=plan.source_root,
                source_paths=plan.source_paths,
                header_paths=plan.header_paths,
                include_dirs=plan.include_dirs,
            ),
            build_type=build_type,
        )
        results.append(result)
        if not result.success:
            break

    return CompiledSteppableBuildSummary(project_path, plans, tuple(results), compiler_selection.warnings)


def format_compiled_steppable_summary(summary: CompiledSteppableBuildSummary) -> str:
    lines = ["Compiled steppable auto-compile summary:"]
    lines.extend(f"  {warning}" for warning in summary.warnings)
    if not summary.plans:
        lines.append("  No C++ compiled steppable resources found.")
        return "\n".join(lines)

    for plan in summary.plans:
        action = "compile" if plan.needs_compile else "skip"
        lines.append(f"  [{action}] {plan.source_path}")
        lines.append(f"         -> {plan.output_path} ({plan.reason})")
        lines.append(f"         metadata -> {plan.metadata_path}")

    for result in summary.results:
        status = "success" if result.success else "failed"
        lines.append(f"  [{status}] {result.output_path}")
        if result.success:
            continue
        log = result.log.strip()
        if log:
            lines.append(log)

    return "\n".join(lines)
