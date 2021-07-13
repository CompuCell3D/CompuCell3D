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

    print(cc3d.get_formatted_version_info())

    cc3d_proj_fname = args.input
    cc3d_proj_fname = cc3d_proj_fname.replace('"', '')
    output_dir = args.output_dir
    output_dir = output_dir.replace('"', '')
    gui_flag = args.gui
    output_frequency = args.output_frequency
    screenshot_output_frequency = args.screenshot_output_frequency
    install_dir = args.install_dir
    install_dir = install_dir.replace('"', '')

    run_script = find_run_script(install_dir=install_dir, gui_flag=gui_flag)

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

        run_single_param_scan_simulation(cc3d_proj_fname=cc3d_proj_fname, run_script=run_script, gui_flag=gui_flag,
                                         current_scan_parameters=current_scan_parameters, output_dir=output_dir,
                                         arg_list=arg_list)

        if stop_scan:
            handle_param_scan_complete(output_dir)
            break


def process_cml():
    cml_parser = argparse.ArgumentParser(description='param_scan_run - Parameter Scan Run Script')
    cml_parser.add_argument('-i', '--input', required=True, action='store',
                            help='path to the CC3D project file (*.cc3d)')
    cml_parser.add_argument('-o', '--output-dir', required=True, action='store',
                            help='path to the output folder to store parameter scan results')
    cml_parser.add_argument('-f', '--output-frequency', required=False, action='store', default=0, type=int,
                            help='simulation snapshot output frequency', dest='output_frequency')
    cml_parser.add_argument('--screenshot-output-frequency', required=False, action='store', default=0, type=int,
                            help='screenshot output frequency')
    cml_parser.add_argument('--gui', required=False, action='store_true', default=False,
                            help='flag indicating whether to use Player or not')
    cml_parser.add_argument('--install-dir', required=True, type=str, help='CC3D install directory')

    args = cml_parser.parse_args()
    return args


def find_run_script(install_dir, gui_flag=False):
    if gui_flag:
        possible_scripts = ['compucell3d.bat', 'compucell3d.command', 'compucell3d.sh']
    else:
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
