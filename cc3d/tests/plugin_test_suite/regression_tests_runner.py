"""
This script facilitates unit/integration testing. Example command looks as follows:

python regression_tests_runner.py --run-command=c:/CompuCell3D-py3-64bit/runScript.bat --output-dir=c:/CompuCell3D_test_output
"""

from cc3d.tests.test_utils.RunSpecs import RunSpecs
from cc3d.tests.test_utils.RunExecutor import RunExecutor
from cc3d import run_script
import argparse
from cc3d.tests.test_utils.common import find_file_in_dir
from os.path import *
import os
import shutil
from glob import glob
import sys
from pathlib import Path
import pandas as pd


def process_cml():
    cml_parser = argparse.ArgumentParser(description="Simulation Tester")
    cml_parser.add_argument(
        "--run-command", nargs="+", type=str, required=False, help="cc3d run script (either RunScript or compucell3d)"
    )
    cml_parser.add_argument("--output-dir", required=True, help="test output dir")

    args = cml_parser.parse_args()

    return args


def main():
    """

    :return:
    """

    simulations_to_run = find_test_run_simulations()

    run_command_list = [sys.executable, run_script.__file__]

    args = process_cml()

    # current_script_dir = dirname(__file__)

    # run_command = ''
    # test_output_root = ''

    # if sys.platform.startswith('win'):
    #     run_command = args.run_command
    #     test_output_root = args.output_dir

    # cc3d_projects = find_file_in_dir(current_script_dir, '*.cc3d')
    cc3d_projects_common_prefix = commonprefix(simulations_to_run)

    rs = RunSpecs()
    rs.run_command = run_command_list
    rs.player_interactive_flag = False
    rs.cc3d_project = ""
    rs.num_steps = 1000
    rs.test_output_root = args.output_dir
    rs.test_output_dir = ""

    # if sys.platform.startswith('win'):
    #     rs.run_command = run_command
    #     rs.player_interactive_flag = False
    #     rs.cc3d_project = ''
    #     rs.num_steps = 1000
    #     rs.test_output_root = test_output_root
    #     rs.test_output_dir = ''

    # clean test_output_dir
    try:
        shutil.rmtree(rs.test_output_root)
    except OSError:
        pass

    try:
        os.makedirs(rs.test_output_root)
    except OSError:
        pass

    # writing to the file list of files to be tested
    with open(join(rs.test_output_root, "cc3d_simulation_test_plan.txt"), "a") as fout:
        for i, cc3d_project in enumerate(simulations_to_run):
            fout.write("{}\n".format(abspath(cc3d_project)))

    error_runs = []

    os.environ["CC3D_TEST_OUTPUT_DIR"] = rs.test_output_root
    os.environ["CC3D_TEST_OUTPUT_SUMMARY"] = join(rs.test_output_root, "test_summary.csv")

    errors_summary_path = Path(rs.test_output_root).joinpath("regression_test_errors.csv")

    for i, cc3d_project in enumerate(simulations_to_run):
        # if Path(cc3d_project).name != "connectivity_elongation_fast.cc3d":
        #     continue
        # if Path(cc3d_project).name != "FocalPointPlasticityCustom.cc3d":
        #     continue

        rs.cc3d_project = cc3d_project
        rs.test_output_dir = relpath(cc3d_project, cc3d_projects_common_prefix)
        run_executor = RunExecutor(run_specs=rs)
        run_executor.run()
        run_status = run_executor.get_run_status()
        if run_status:
            error_tuple = (rs.cc3d_project, run_status)
            error_runs.append(error_tuple)

            error_df = pd.DataFrame({"file_name": [rs.cc3d_project], "status": [run_status]})

            cc3d_simulation_tests_output_summary_df = pd.DataFrame()
            if errors_summary_path.exists():
                cc3d_simulation_tests_output_summary_df = pd.read_csv(errors_summary_path)

            cc3d_simulation_tests_output_summary_df = pd.concat(
                [cc3d_simulation_tests_output_summary_df, error_df], ignore_index=True
            )

            cc3d_simulation_tests_output_summary_df.to_csv(errors_summary_path, index=False)

    if not len(error_runs):
        print("\n-----------------ALL SIMULATIONS RUN SUCCESSFULLY----------------------\n")

    else:
        print("\n-----------------THERE WERE ERRORS IN THE SIMULATIONS----------------------\n")

        for error_run in error_runs:
            print(error_run)


def find_test_run_simulations():
    """
    Locates *.cc3d projects that are regression test runs
    :return:
    """

    test_run_dirs_to_process = [x[0] for x in os.walk(Path(__file__).parent) if x[0].endswith("_test_run")]
    test_run_simulations = []
    for test_run_dir in test_run_dirs_to_process:
        simulation_match = glob(str(Path(test_run_dir).joinpath("*.cc3d")))
        try:
            test_run_simulations.append(simulation_match[0])
        except IndexError:
            print(f"Could not locate valid *.cc3d file in {test_run_dir}")
            continue

    return test_run_simulations


if __name__ == "__main__":
    main()
