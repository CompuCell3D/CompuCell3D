import argparse
import traceback
import cc3d
from os.path import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.RollbackImporter import RollbackImporter

"""
-i d:\CC3DProjects\bac_mac_restart_100\bacterium_macrophage_2D_steering.cc3d -f 10 -fr 100 --restart-multiple-snapshots
"""

def process_cml(known_args=None):
    """

    :return:
    """
    cml_parser = argparse.ArgumentParser(description='CompuCell3D Player 5')
    cml_parser.add_argument('-i', '--input', required=True, action='store',
                            help='path to the CC3D project file (*.cc3d)')

    cml_parser.add_argument('-c', '--output-file-core-name', required=False, action='store', default='Step',
                            help='core name for vtk files.')

    cml_parser.add_argument('--current-dir', required=False, action='store',
                            help='path to current directory')

    cml_parser.add_argument('-o', '--output-dir', required=False, action='store',
                            help='path to the output folder to store simulation results')

    cml_parser.add_argument('-f', '--output-frequency', required=False, action='store', default=0, type=int,
                            help='simulation snapshot output frequency')

    cml_parser.add_argument('-fs', '--screenshot-output-frequency', required=False, action='store', default=0, type=int,
                            help='screenshot output frequency')

    cml_parser.add_argument('-fr', '--restart-snapshot-frequency', required=False, action='store', default=0, type=int,
                            help='restart snapshot output frequency')

    cml_parser.add_argument('--restart-multiple-snapshots', required=False, action='store_true', default=False,
                            help='turns on storing of multiple restart snapshots')

    cml_parser.add_argument('--parameter-scan-iteration', required=False, type=str, default='',
                            help='optional argument that specifies parameter scan iteration - used to enable steppables'
                                 'to access current param scan iteration number')

    cml_parser.add_argument('--log-level', required=False, type=str, default='',
                            choices=['', 'FATAL', 'CRITICAL', 'ERROR',
                                     'WARNING', 'NOTICE', 'INFORMATION',
                                     'DEBUG', 'TRACE', 'CURRENT'],
                            help='optional argument that specifies log level: allowed values are:'
                                 'FATAL, CRITICAL, ERROR, WARNING, '
                                 'NOTICE, INFORMATION, DEBUG, TRACE CURRENT')

    cml_parser.add_argument('--log-to-file', required=False, action='store_true', default=False,
                            help='optional argument that specifies if log should be saved to a file')


    return cml_parser.parse_args(args=known_args)


def handle_error():
    """

    :return:
    """

    tb = traceback.format_exc()

    formatted_lines = tb.splitlines()
    error_description_line = formatted_lines[-1]

    traceback_text = "Error: " + error_description_line + "\n"
    traceback_text += tb

    simthread = CompuCellSetup.persistent_globals.simthread

    if simthread is not None:
        # simthread.emitErrorOccured('Python Error', tb)
        simthread.emitErrorFormatted(traceback_text)

def main(args:argparse.Namespace=None):
    if not args:
        args = process_cml()

    print(cc3d.get_formatted_version_info())

    cc3d_sim_fname = args.input
    output_frequency = args.output_frequency
    screenshot_output_frequency = args.screenshot_output_frequency
    restart_snapshot_frequency = args.restart_snapshot_frequency
    restart_multiple_snapshots = args.restart_multiple_snapshots

    output_dir = args.output_dir
    output_file_core_name = args.output_file_core_name

    persistent_globals = cc3d.CompuCellSetup.persistent_globals

    rollbackImporter = RollbackImporter()

    current_dir = args.current_dir if args.current_dir else ''
    cc3d_sim_fname_abs = join(current_dir, cc3d_sim_fname)

    persistent_globals.simulation_file_name = cc3d_sim_fname_abs
    persistent_globals.output_frequency = output_frequency
    persistent_globals.screenshot_output_frequency = screenshot_output_frequency
    persistent_globals.set_output_dir(output_dir)
    persistent_globals.output_file_core_name = output_file_core_name
    persistent_globals.restart_snapshot_frequency = restart_snapshot_frequency
    persistent_globals.restart_multiple_snapshots = restart_multiple_snapshots
    persistent_globals.parameter_scan_iteration = args.parameter_scan_iteration

    if args.log_level:
        persistent_globals.log_level = "LOG_" + args.log_level
    else:
        persistent_globals.log_level = "LOG_CURRENT"

    persistent_globals.log_to_file = args.log_to_file

    run_cc3d_project(cc3d_sim_fname=cc3d_sim_fname_abs)

    rollbackImporter.uninstall()


if __name__ == '__main__':
    main()