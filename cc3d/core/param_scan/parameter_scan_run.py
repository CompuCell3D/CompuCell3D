"""
this is a main script that facilitates parameter scan runs. It launches either command line only run script or a player
and repeatedly runs the same simulation with different parameters. To test parameter scans it is best to work in python
environment that has all the components - core code + player installed. Otherwise, you might get cryptic messages like
cc3d.player5 module not found etc...
"""

import argparse
import sys
from pathlib import Path
from typing import Union
import cc3d
from os.path import commonprefix
from cc3d.core.param_scan.parameter_scan_utils import copy_project_to_output_folder
from cc3d.core.param_scan.parameter_scan_utils import param_scan_status_path
from cc3d.core.param_scan.parameter_scan_utils import create_param_scan_status
from cc3d.core.param_scan.parameter_scan_utils import cc3d_proj_pth_in_output_dir
from cc3d.core.param_scan.parameter_scan_utils import fetch_next_set_of_scan_parameters
from cc3d.core.param_scan.parameter_scan_utils import run_single_param_scan_simulation
from cc3d.core.param_scan.parameter_scan_utils import param_scan_complete_signal
from cc3d.core.param_scan.parameter_scan_utils import handle_param_scan_complete
from cc3d.core.param_scan.parameter_scan_utils import read_parameters_from_param_scan_status_file
from cc3d.core.param_scan.parameter_scan_utils import ParamScanStop
from cc3d.core.filelock import FileLock
from cc3d.core.filelock import FileLockException


def main():
    args = process_cml()

    cc3d_proj_fname = args.input
    cc3d_proj_fname = cc3d_proj_fname.replace('"', '')
    output_dir = args.output_dir
    output_dir = output_dir.replace('"', '')
    output_frequency = args.output_frequency
    screenshot_output_frequency = args.screenshot_output_frequency
    install_dir = args.install_dir
    install_dir = install_dir.replace('"', '')

    execute_scan(cc3d_proj_fname=cc3d_proj_fname,
                 output_dir=output_dir,
                 output_frequency=output_frequency,
                 screenshot_output_frequency=screenshot_output_frequency,
                 gui_flag=args.gui)

def execute_scan(cc3d_proj_fname: str,
                 output_dir: str,
                 output_frequency: int,
                 screenshot_output_frequency: int,
                 gui_flag: bool = False):
    """
    Executes parameter scan

    :param cc3d_proj_fname: absolute path to cc3d project file
    :type cc3d_proj_fname: str
    :param output_dir: absolute path to output directory
    :type output_dir: str
    :param output_frequency: output frequency
    :type output_frequency: int
    :param screenshot_output_frequency: screenshot output frequency
    :type screenshot_output_frequency: int
    # :param run_script: absolute path to single run execution script
    # :type run_script: str
    :param gui_flag: optional flag for gui-based execution; launches will pass additional argument '--exit-when-done'
    :type gui_flag: bool
    :return: None
    """

    print(cc3d.get_formatted_version_info())

    param_scan_complete_pth = param_scan_complete_signal(output_dir=output_dir)

    if param_scan_complete_pth.exists():
        handle_param_scan_complete(output_dir)

    try:
        prepare_param_scan_folder(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)
    except FileLockException:

        print('There exists a {lock_file} that prevents param scan from running. '
              'Please remove this file and start again'.format(
            lock_file=Path(output_dir).joinpath('param_scan_status.lock')))

        sys.exit(1)

    stop_scan = False

    while True:
        try:
            current_scan_parameters = fetch_next_set_of_scan_parameters(output_dir=output_dir)
            print('immediate ', current_scan_parameters)
        except FileLockException:

            print('There exists a {lock_file} that prevents param scan from running. '
                  'Please remove this file and start again'.format(
                lock_file=Path(output_dir).joinpath('param_scan_status.lock')))

            break

        except ParamScanStop:
            stop_scan = True
            current_scan_parameters, _ = read_parameters_from_param_scan_status_file(output_dir=output_dir)

        # event with ParamScanStop signal we run the last simulation in the param scan. After this last run
        # param_scan.complete.signal will get written to the disk
        arg_list = [
            f'--output-frequency={output_frequency}',
            f'--screenshot-output-frequency={screenshot_output_frequency}'
        ]

        run_single_param_scan_simulation(cc3d_proj_fname=cc3d_proj_fname, gui_flag=gui_flag,
                                         current_scan_parameters=current_scan_parameters, output_dir=output_dir,
                                         arg_list=arg_list)


        if stop_scan:
            handle_param_scan_complete(output_dir)
            break


class ParamScanArgumentParser(argparse.ArgumentParser):
    """Argument parsers for parameter scans"""

    def __init__(self):
        super().__init__(description='param_scan_run - Parameter Scan Run Script')

        self.add_argument('-i', '--input', required=True, action='store',
                          help='path to the CC3D project file (*.cc3d)')
        self.add_argument('-o', '--output-dir', required=True, action='store',
                          help='path to the output folder to store parameter scan results')
        self.add_argument('-f', '--output-frequency', required=False, action='store', default=0, type=int,
                          help='simulation snapshot output frequency', dest='output_frequency')
        self.add_argument('--screenshot-output-frequency', required=False, action='store', default=0, type=int,
                          help='screenshot output frequency')
        self.add_argument('--install-dir', required=True, type=str, help='CC3D install directory')
        # Legacy support
        self.add_argument('--gui', required=False, action='store_true', default=False,
                          help='flag indicating whether to use Player or not')


def process_cml():
    return ParamScanArgumentParser().parse_args()


def find_run_script(install_dir):
    possible_scripts = ['runScript.bat', 'runScript.command', 'runScript.sh']

    for script_name in possible_scripts:
        full_script_path = Path(install_dir).joinpath(script_name)
        if full_script_path.exists():
            return str(full_script_path)

    raise FileNotFoundError('Could not find run script')


def ensure_output_folder_outside_simulation_project(cc3d_proj_fname: Union[str, Path],
                                                    output_dir: Union[str, Path]) -> None:
    """
    Ensures output folder is outside folder with simulation files - to avoid excessive, recursive file copies.
    Raises runtime error when user tries to output param scan inside project folder
    :param cc3d_proj_fname: path to .cc3d project
    :param output_dir: path to parameter scan output dir
    :return:
    """

    proj_fname = Path(cc3d_proj_fname)
    proj_folder = proj_fname.parent
    out_dir = Path(output_dir)
    prefix = commonprefix([proj_folder, out_dir])
    if Path(prefix) == Path(proj_folder):
        raise RuntimeError(f'You are trying to write parameter scan output to the Simulation folder: {out_dir}. '
                           f'This is not allowed'
                           f'Please choose folder outside Simulation folder')


def prepare_param_scan_folder(cc3d_proj_fname: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Prepares parameter scan folder - copies necessary files . Only one instance can run this function it is
    lock-protected.
    :param cc3d_proj_fname: path to .cc3d project
    :param output_dir: path to parameter scan output dir
    :return:
    """

    ensure_output_folder_outside_simulation_project(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with FileLock(Path(output_dir).joinpath('param_scan_status.lock')):
        if not param_scan_status_path(output_dir).exists():
            copy_project_to_output_folder(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

            cc3d_proj_target = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

            create_param_scan_status(cc3d_proj_target, output_dir=output_dir)


if __name__ == '__main__':
    main()
