from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationDataHandler
from cc3d.core.CompiledSteppableCompilerSettings import compiler_selection_from_settings
from cc3d.scripts.compile_steppable import (
    CompileResult,
    build_output_path,
    compile_steppable_source,
    resolve_kernel_include_root,
)


@dataclass(frozen=True)
class CompiledSteppableBuildPlan:
    source_path: Path
    output_path: Path
    needs_compile: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
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

    plans: list[CompiledSteppableBuildPlan] = []
    for source_path in source_paths:
        output_path = build_output_path(source_path)
        if force:
            plans.append(CompiledSteppableBuildPlan(source_path, output_path, True, "forced"))
            continue
        if not source_path.exists():
            plans.append(CompiledSteppableBuildPlan(source_path, output_path, True, "source missing"))
            continue
        if not output_path.exists():
            plans.append(CompiledSteppableBuildPlan(source_path, output_path, True, "output missing"))
            continue

        output_mtime = output_path.stat().st_mtime
        if source_path.stat().st_mtime > output_mtime:
            plans.append(CompiledSteppableBuildPlan(source_path, output_path, True, "source newer than output"))
            continue
        if kernel_header.exists() and kernel_header.stat().st_mtime > output_mtime:
            plans.append(CompiledSteppableBuildPlan(source_path, output_path, True, "kernel header newer than output"))
            continue

        plans.append(CompiledSteppableBuildPlan(source_path, output_path, False, "up to date"))

    return plans


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

    for result in summary.results:
        status = "success" if result.success else "failed"
        lines.append(f"  [{status}] {result.output_path}")
        if result.success:
            continue
        log = result.log.strip()
        if log:
            lines.append(log)

    return "\n".join(lines)
