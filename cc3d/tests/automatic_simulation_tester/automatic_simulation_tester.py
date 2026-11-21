"""Run every CC3D project below a directory in a separate Python process."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_PLAYER_MODULE = "cc3d.player5"


@dataclass(frozen=True)
class SimulationResult:
    project: Path
    returncode: int

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def discover_projects(simulation_folder: Path) -> list[Path]:
    """Return projects below *simulation_folder* in stable order."""
    return sorted(path for path in simulation_folder.rglob("*.cc3d") if path.is_file())


def build_command(
    project: Path,
    python_interpreter: str,
    runner: str,
    max_mcs: int,
    player_module: str = DEFAULT_PLAYER_MODULE,
    output_dir: Path | None = None,
) -> list[str]:
    """Build the subprocess command for one project."""
    if max_mcs < 0:
        raise ValueError("max_mcs must be non-negative")
    if runner == "run-script":
        runner = "run_script"
    if runner not in {"player", "run_script"}:
        raise ValueError(f"unsupported runner: {runner}")

    module = player_module if runner == "player" else "cc3d.run_script"
    command = [python_interpreter, "-m", module]
    if runner == "player":
        command.extend([
            "--input", str(project),
            "--numSteps", str(max_mcs),
            "--exit-when-done",
            "--noOutput",
        ])
    else:
        command.extend(["--input", str(project), "--max-mcs", str(max_mcs)])

    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    return command


def run_projects(
    projects: Sequence[Path],
    python_interpreter: str,
    runner: str,
    max_mcs: int,
    player_module: str = DEFAULT_PLAYER_MODULE,
    output_dir: Path | None = None,
) -> list[SimulationResult]:
    """Run projects sequentially and return one result per project."""
    results: list[SimulationResult] = []
    for project in projects:
        project_output_dir = safe_output_dir(project, output_dir)
        command = build_command(project, python_interpreter, runner, max_mcs, player_module, project_output_dir)
        print(f"RUNNING {project}", flush=True)
        completed = subprocess.run(
            command,
            cwd=project.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        # Some CC3D launchers can print an unhandled traceback but still return
        # zero. Treat that as a failed simulation rather than a false positive.
        has_traceback = "Traceback (most recent call last):" in (
            completed.stdout + completed.stderr
        )
        returncode = completed.returncode if completed.returncode != 0 or not has_traceback else 1
        result = SimulationResult(project=project, returncode=returncode)
        results.append(result)
        print(f"{'PASSED' if result.passed else 'FAILED'} {project}", flush=True)
    return results


def safe_output_dir(project: Path, output_root: Path | None = None) -> Path:
    """Return an output directory that is not the project folder or its child."""
    root = (output_root or Path.home() / "CC3DWorkspace").resolve()
    project_folder = project.resolve().parent
    output_dir = root / project.stem

    if output_dir == project_folder or output_dir.is_relative_to(project_folder):
        fallback_root = Path(tempfile.mkdtemp(prefix="cc3d_simulation_output_"))
        output_dir = fallback_root / project.stem

    return output_dir


def default_report_path(simulation_folder: Path, now: datetime | None = None) -> Path:
    """Return the default report path in the caller's current directory."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    folder_name = simulation_folder.name or "simulations"
    return Path.cwd() / f"cc3d_simulation_report_{folder_name}_{timestamp}.txt"


def write_report(
    report_path: Path,
    simulation_folder: Path,
    python_interpreter: str,
    runner: str,
    max_mcs: int,
    results: Sequence[SimulationResult],
) -> None:
    """Write a human-readable run report."""
    failed = sum(not result.passed for result in results)
    lines = [
        "CompuCell3D Automatic Simulation Tester Report",
        f"Simulation folder: {simulation_folder}",
        f"Python interpreter: {python_interpreter}",
        f"Runner: {runner}",
        f"Maximum MCS: {max_mcs}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Results:",
    ]
    lines.extend(
        f"{'PASSED' if result.passed else 'FAILED'}\treturncode={result.returncode}\t{result.project}"
        for result in results
    )
    lines.extend(["", f"Summary: {len(results) - failed} passed, {failed} failed"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_folder", type=Path)
    parser.add_argument("--python", dest="python_interpreter", default=sys.executable,
                        help="Python interpreter used to run CC3D (default: current interpreter)")
    parser.add_argument("--runner", choices=("player", "run_script", "run-script"), default="run_script")
    parser.add_argument("--max-mcs", type=int, default=200)
    parser.add_argument("--player-module", default=DEFAULT_PLAYER_MODULE,
                        help=f"Player module for --runner player (default: {DEFAULT_PLAYER_MODULE})")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path,
                        help="report path (default: current directory with an automatic filename)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    simulation_folder = args.simulation_folder.resolve()
    if not simulation_folder.is_dir():
        print(f"Simulation folder does not exist or is not a directory: {simulation_folder}", file=sys.stderr)
        return 2

    projects = discover_projects(simulation_folder)
    if not projects:
        print(f"No .cc3d projects found below {simulation_folder}", file=sys.stderr)
        return 2

    results = run_projects(
        projects=projects,
        python_interpreter=args.python_interpreter,
        runner=args.runner,
        max_mcs=args.max_mcs,
        player_module=args.player_module,
        output_dir=args.output_dir,
    )
    report_path = args.report or default_report_path(simulation_folder)
    write_report(
        report_path=report_path,
        simulation_folder=simulation_folder,
        python_interpreter=args.python_interpreter,
        runner=args.runner,
        max_mcs=args.max_mcs,
        results=results,
    )
    failed = sum(not result.passed for result in results)
    print(f"REPORT: {report_path}")
    print(f"SUMMARY: {len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
