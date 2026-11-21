from pathlib import Path
from datetime import datetime

from cc3d.tests.automatic_simulation_tester.automatic_simulation_tester import (
    build_command,
    default_report_path,
    discover_projects,
    safe_output_dir,
)


def test_discover_projects_is_recursive_and_sorted(tmp_path):
    (tmp_path / "b" / "nested").mkdir(parents=True)
    (tmp_path / "a.cc3d").touch()
    (tmp_path / "b" / "nested" / "z.cc3d").touch()
    (tmp_path / "b" / "ignore.txt").touch()

    assert discover_projects(tmp_path) == [
        tmp_path / "a.cc3d",
        tmp_path / "b" / "nested" / "z.cc3d",
    ]


def test_build_run_script_command():
    project = Path("/sim/example.cc3d")
    assert build_command(project, "/env/bin/python", "run_script", 200) == [
        "/env/bin/python", "-m", "cc3d.run_script",
        "--input", "/sim/example.cc3d", "--max-mcs", "200",
    ]


def test_build_player_command():
    project = Path("/sim/example.cc3d")
    assert build_command(project, "/env/bin/python", "player", 200, "cc3d.player") == [
        "/env/bin/python", "-m", "cc3d.player",
        "--input", "/sim/example.cc3d", "--numSteps", "200",
        "--exit-when-done", "--noOutput",
    ]


def test_default_report_path_uses_current_directory_and_minute_timestamp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_path = default_report_path(
        Path("/simulations/my_suite"),
        datetime(2026, 11, 21, 13, 42, 59),
    )

    assert report_path == tmp_path / "cc3d_simulation_report_my_suite_20261121_1342.txt"


def test_safe_output_dir_is_outside_project_folder(tmp_path):
    project = tmp_path / "suite" / "example.cc3d"
    project.parent.mkdir()

    output_dir = safe_output_dir(project, project.parent / "output")

    assert not output_dir.is_relative_to(project.parent)
    assert output_dir == (project.parent / "output" / "example").resolve()
