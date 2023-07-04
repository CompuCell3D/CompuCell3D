"""
This iscript facilitates unit/integration testing. Example command looks as follows:

python simulation_tester.py --run-command=c:/CompuCell3D-py3-64bit/runScript.bat --output-dir=c:/CompuCell3D_test_output
"""
from cc3d.tests.test_utils.RunSpecs import RunSpecs
from cc3d.tests.test_utils.RunExecutor import RunExecutor
import argparse
from cc3d.tests.test_utils.common import find_file_in_dir
from os.path import *
import os
import shutil
import sys
from cc3d import run_script
from pathlib import Path


def process_cml():
    cml_parser = argparse.ArgumentParser(description='Simulation Tester')
    cml_parser.add_argument('--run-command', required=False,
                            help='cc3d run script (either RunScript or compucell3d)', default="")
    cml_parser.add_argument('--output-dir', required=True,
                            help='test output dir')

    args = cml_parser.parse_args()

    return args


def main():
    """

    :return:
    """
    args = process_cml()

    current_script_dir = dirname(__file__)

    # run_command = args.run_command
    run_command_list = [sys.executable, run_script.__file__]
    test_output_root = args.output_dir

    # testing only PDE_solvers for now
    cc3d_projects = find_file_in_dir(Path(current_script_dir).joinpath("pde_solvers"), '*.cc3d')
    cc3d_projects_common_prefix = commonprefix(cc3d_projects)

    rs = RunSpecs()
    # rs.run_command = run_command
    rs.run_command = run_command_list
    rs.player_interactive_flag = False
    rs.cc3d_project = ''
    rs.num_steps = 1000
    rs.test_output_root = test_output_root
    rs.test_output_dir = ''

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
    with open(join(rs.test_output_root, 'cc3d_simulation_test_plan.txt'), 'a') as fout:
        for i, cc3d_project in enumerate(cc3d_projects):
            fout.write('{}\n'.format(abspath(cc3d_project)))

    error_runs = []

    os.environ['CC3D_TEST_OUTPUT_DIR'] = rs.test_output_root
    os.environ['CC3D_TEST_OUTPUT_SUMMARY'] = join(rs.test_output_root, 'test_summary.csv')

    for i, cc3d_project in enumerate(cc3d_projects):
        print(f"cc3d_project={cc3d_project}")
        rs.cc3d_project = cc3d_project
        rs.test_output_dir = relpath(cc3d_project, cc3d_projects_common_prefix)
        run_executor = RunExecutor(run_specs=rs)
        run_executor.run()
        run_status = run_executor.get_run_status()
        if run_status:
            error_tuple = (rs.cc3d_project, run_status)
            error_runs.append(error_tuple)
            with open(join(rs.test_output_root, 'cc3d_simulation_tests.txt'), 'a') as fout:
                fout.write('{}\n'.format(error_tuple))

    if not len(error_runs):
        print()
        print('-----------------ALL SIMULATIONS RUN SUCCESSFULLY----------------------')
        print()

    else:
        print()
        print('-----------------THERE WERE ERRORS IN THE SIMULATIONS----------------------')
        print()

        for error_run in error_runs:
            print(error_run)


if __name__ == '__main__':
    main()

