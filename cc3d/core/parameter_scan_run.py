import argparse
from pathlib import Path
from cc3d.core.parameter_scan_utils import copy_project_to_output_folder
from cc3d.core.parameter_scan_utils import create_param_scan_status
from cc3d.core.parameter_scan_utils import cc3d_proj_pth_in_output_dir
from cc3d.core.parameter_scan_utils import fetch_next_set_of_scan_parameters
from cc3d.core.parameter_scan_utils import run_single_param_scan_simulation
from cc3d.core.parameter_scan_utils import param_scan_complete_signal
from cc3d.core.parameter_scan_utils import handle_param_scan_complete
from cc3d.core.filelock import FileLockException


def process_cml():
    cml_parser = argparse.ArgumentParser(description='param_scan_run - Parameter Scan Run Script')
    cml_parser.add_argument('-i', '--input', required=True, action='store',
                            help='path to the CC3D project file (*.cc3d)')
    cml_parser.add_argument('-o', '--output', required=True, action='store',
                            help='path to the output folder to store parameter scan results')
    cml_parser.add_argument('-f', '--outputFrequency', required=False, action='store', default=1, type=int,
                            help='simulation snapshot output frequency')

    args = cml_parser.parse_args()
    return args


if __name__ == '__main__':
    args = process_cml()

    cc3d_proj_fname = args.input
    output_dir = args.output

    param_scan_complete_pth = param_scan_complete_signal(output_dir=output_dir)

    if param_scan_complete_pth.exists():
        handle_param_scan_complete(output_dir)

    copy_success = copy_project_to_output_folder(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

    cc3d_proj_target = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

    create_param_scan_status(cc3d_proj_target, output_dir=output_dir)

    while True:
        try:
            current_scan_parameters = fetch_next_set_of_scan_parameters(output_dir=output_dir)
        except FileLockException:

            print('There exists a {lock_file} that prevents param scan from running. '
                  'Please remove this file and start again'.format(
                lock_file=Path(output_dir).joinpath('param_scan_status.lock')))

            break

        except StopIteration:
            handle_param_scan_complete(output_dir)

            break

        run_single_param_scan_simulation(current_scan_parameters=current_scan_parameters)
