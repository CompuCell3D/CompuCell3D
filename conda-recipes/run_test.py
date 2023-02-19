from cc3d.tests import simulation_tester
from cc3d.tests.plugin_test_suite import regression_tests_runner
from cc3d import cc3d_scripts_path
import os
import subprocess
import sys
from pathlib import Path

test_plugin_dir = Path.cwd().joinpath('tests_regression')
test_plugin_dir.mkdir(exist_ok=True, parents=True)


test_dir = os.path.join(os.getcwd(), 'tests')

if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

exec_dir = os.path.dirname(simulation_tester.__file__)

if sys.platform.startswith('win'):
    ext = '.bat'
elif sys.platform.startswith('darwin'):
    ext = '.command'
else:
    ext = '.sh'
# Handle variants: if cc3d was not built as standalone, then script names are prefixed with "cc3d_"
run_script = os.path.join(cc3d_scripts_path, 'cc3d_runScript' + ext)
if not os.path.isfile(run_script):
    run_script = os.path.join(cc3d_scripts_path, 'runScript' + ext)

# If something goes wrong in a test, then this file is created during post-processing
# We'll use it as a signal to signal failure
# cc3d_simulation_tests.txt - reports non-zero exit code but ignores regression errors
# we will need to address non-zero exit codes as well but for now lets focus on regression errors
# fail_file = os.path.join(test_dir, 'cc3d_simulation_tests.txt')

# test_regression_errors.csv - reports actual regression testing errors
fail_file = os.path.join(test_dir, 'test_regression_errors.csv')

plugin_test_suite_fail_file = test_plugin_dir.joinpath('regression_test_errors.csv')


def main():
    print()
    print('-----------------PERFORMING CC3D TESTS----------------------')
    print('Generating test results data in', test_dir)
    print()

    # from cc3d.cpp import CompuCell
    # sys.exit(0)
    # run pde solvers regression suite
    subprocess.check_call(args=['python', simulation_tester.__file__,
                                # f'--run-command={run_script}',
                                f'--output-dir={test_dir}'],
                          cwd=exec_dir)
    if os.path.isfile(fail_file):
        sys.exit(1)

    # run plugin regression suite
    subprocess.check_call(args=['python', regression_tests_runner.__file__,
                                # f'--run-command={run_script}',
                                f'--output-dir={test_plugin_dir}'],
                          cwd=exec_dir)

    if plugin_test_suite_fail_file.exists():
        sys.exit(1)


if __name__ == '__main__':
    main()
