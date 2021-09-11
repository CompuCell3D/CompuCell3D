from pathlib import Path
from cc3d.core.param_scan.parameter_scan_run import ParamScanArgumentParser, execute_scan


def find_run_script(install_dir):
    possible_scripts = ['compucell3d.bat', 'compucell3d.command', 'compucell3d.sh']

    for script_name in possible_scripts:
        full_script_path = Path(install_dir).joinpath(script_name)
        if full_script_path.exists():
            return str(full_script_path)

    raise FileNotFoundError('Could not find run script')


def main():
    args = ParamScanArgumentParser().parse_args()

    cc3d_proj_fname = args.input
    cc3d_proj_fname = cc3d_proj_fname.replace('"', '')
    output_dir = args.output_dir
    output_dir = output_dir.replace('"', '')
    output_frequency = args.output_frequency
    screenshot_output_frequency = args.screenshot_output_frequency
    install_dir = args.install_dir
    install_dir = install_dir.replace('"', '')

    run_script = find_run_script(install_dir=install_dir)

    execute_scan(cc3d_proj_fname=cc3d_proj_fname,
                 output_dir=output_dir,
                 output_frequency=output_frequency,
                 screenshot_output_frequency=screenshot_output_frequency,
                 run_script=run_script,
                 gui_flag=True)
