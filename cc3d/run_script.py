import argparse
import traceback
import cc3d
from os.path import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.CompiledSteppableAutoCompiler import (
    clean_project_steppables,
    compile_project_steppables,
    format_compiled_steppable_summary,
)
from cc3d.core.CompiledSteppableCompilerSettings import auto_compile_enabled
from cc3d.core.DefaultSettingsData import SETTINGS_FOLDER
from cc3d.core.RollbackImporter import RollbackImporter


def set_global_settings_dir_name(global_settings_dir_name):
    """
    Overrides the global settings directory name before settings are loaded.
    """
    import cc3d.core.DefaultSettingsData as default_settings_data
    import cc3d.core.Configuration.SettingUtils as setting_utils

    default_settings_data.SETTINGS_FOLDER = global_settings_dir_name
    setting_utils.SETTINGS_FOLDER = global_settings_dir_name
    CompuCellSetup.persistent_globals._configuration = None


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

    cml_parser.add_argument(
        '--execute-step-at-mcs-0',
        action='store_true',
        default=False,
        help='Execute step at MCS 0 when this flag is present.'
    )

    cml_parser.add_argument('--log-level', required=False, type=str, default='',
                            choices=['', 'FATAL', 'CRITICAL', 'ERROR',
                                     'WARNING', 'NOTICE', 'INFORMATION',
                                     'DEBUG', 'TRACE', 'CURRENT'],
                            help='optional argument that specifies log level: allowed values are:'
                                 'FATAL, CRITICAL, ERROR, WARNING, '
                                 'NOTICE, INFORMATION, DEBUG, TRACE CURRENT')

    cml_parser.add_argument('--log-to-file', required=False, action='store_true', default=False,
                            help='optional argument that specifies if log should be saved to a file')

    cml_parser.add_argument('--global-settings-dir-name', required=False, action='store',
                            default=SETTINGS_FOLDER,
                            help='directory name under the user home directory for global settings files')

    cml_parser.add_argument('--no-compile-steppables', required=False, action='store_true', default=False,
                            help='skip automatic compilation of C++ steppable resources listed in the .cc3d file')

    cml_parser.add_argument('--force-compile-steppables', required=False, action='store_true', default=False,
                            help='compile C++ steppable resources even when the output library is up to date')

    cml_parser.add_argument('--clean-compiled-steppables', required=False, action='store_true', default=False,
                            help='remove generated artifacts for all C++ steppable resources and exit')

    cml_parser.add_argument('--clean-and-compile-steppables', required=False, action='store_true', default=False,
                            help='remove generated artifacts, force-compile all C++ steppables, then run')

    cml_parser.add_argument('--steppable-include-root', required=False, action='store',
                            help='include root containing cc3d/kernel.h for compiled steppables')

    cml_parser.add_argument('--steppable-repo', required=False, action='store',
                            help='CompuCell3D repository path used as a source-tree fallback for cc3d/kernel.h')

    cml_parser.add_argument('--steppable-msvc-version', required=False, action='store',
                            help='Windows only: Visual Studio major version for compiling steppables, e.g. 16, 17, 18')

    cml_parser.add_argument('--steppable-macos-toolchain', required=False, action='store',
                            choices=('apple', 'conda'), default=None,
                            help='macOS only: compiler toolchain for compiled steppables')

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

    set_global_settings_dir_name(getattr(args, 'global_settings_dir_name', SETTINGS_FOLDER))

    print(cc3d.get_formatted_version_info(), flush=True)

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

    if args.clean_compiled_steppables and args.clean_and_compile_steppables:
        raise ValueError('--clean-compiled-steppables and --clean-and-compile-steppables are mutually exclusive')
    if args.clean_compiled_steppables and args.force_compile_steppables:
        raise ValueError('--clean-compiled-steppables cannot be combined with --force-compile-steppables')
    if args.clean_and_compile_steppables and args.no_compile_steppables:
        raise ValueError('--clean-and-compile-steppables cannot be combined with --no-compile-steppables')

    if args.clean_compiled_steppables:
        removed_paths = clean_project_steppables(cc3d_sim_fname_abs)
        print('Compiled steppable cleanup:')
        if removed_paths:
            for path in removed_paths:
                print(f'  [removed] {path}')
        else:
            print('  No compiled steppable artifacts found.')
        return

    if args.clean_and_compile_steppables:
        removed_paths = clean_project_steppables(cc3d_sim_fname_abs)
        print('Compiled steppable cleanup:')
        if removed_paths:
            for path in removed_paths:
                print(f'  [removed] {path}')
        else:
            print('  No compiled steppable artifacts found.')

    persistent_globals.execute_step_at_mcs_0 = args.execute_step_at_mcs_0
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

    force_compile = args.force_compile_steppables or args.clean_and_compile_steppables
    if not args.no_compile_steppables and (force_compile or auto_compile_enabled()):
        compile_summary = compile_project_steppables(
            cc3d_project_path=cc3d_sim_fname_abs,
            repo_path=args.steppable_repo,
            include_root=args.steppable_include_root,
            force=force_compile,
            macos_toolchain=args.steppable_macos_toolchain,
            msvc_version=args.steppable_msvc_version,
        )
        print(format_compiled_steppable_summary(compile_summary), flush=True)
        if not compile_summary.success:
            raise RuntimeError("Automatic compilation of C++ steppables failed.")

    run_cc3d_project(cc3d_sim_fname=cc3d_sim_fname_abs)

    rollbackImporter.uninstall()


if __name__ == '__main__':
    main()
