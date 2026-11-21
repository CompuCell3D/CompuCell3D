# Automatic Simulation Tester

`automatic_simulation_tester.py` recursively finds all `.cc3d` project files below a directory and runs each project in a separate Python process.

The tester uses the current Python interpreter by default. Use `--python` when the CC3D installation is in another environment.

## Run with `cc3d.run_script`

This is the default runner:

```bash
python automatic_simulation_tester.py /path/to/simulations \
    --max-mcs 200
```

Equivalent command using an explicit interpreter:

```bash
python automatic_simulation_tester.py /path/to/simulations \
    --python /path/to/cc3d/environment/bin/python \
    --runner run_script \
    --max-mcs 200
```

Each project is run as:

```text
python -m cc3d.run_script --input PROJECT.cc3d --max-mcs 200
```

`--max-mcs` overrides the number of MCS configured in the project.

## Run with Player

Player runs use Player's `--numSteps`, `--exit-when-done`, and `--noOutput` options:

```bash
python automatic_simulation_tester.py /path/to/simulations \
    --runner player \
    --player-module cc3d.player5 \
    --max-mcs 200
```

If the installed Player module is `cc3d.player`, use:

```bash
--player-module cc3d.player
```

## Options

```text
simulation_folder       Root folder searched recursively for .cc3d files
--python PATH            Python interpreter used for child processes
--runner NAME            run_script (default) or player
--max-mcs N              Number of MCS to run (default: 200)
--player-module NAME    Player module (default: cc3d.player5)
--output-dir PATH       Root for per-project output directories (default: ~/CC3DWorkspace)
```

Projects run sequentially. The tester continues after a failed project and prints a summary at the end. It exits with status `0` only when every project succeeds; it exits with status `1` if any project fails and status `2` for invalid input or when no projects are found.

After the run, a text report is written by default to the current working directory. Its name is:

```text
cc3d_simulation_report_<simulation_folder_name>_<YYYYMMDD_HHMM>.txt
```

For example:

```text
cc3d_simulation_report_my_suite_20261121_1342.txt
```

Use `--report PATH` to choose a different report location.

Run the script from the repository root, or provide its full path:

```bash
python cc3d/tests/automatic_simulation_tester/automatic_simulation_tester.py /path/to/simulations --max-mcs 40
```
