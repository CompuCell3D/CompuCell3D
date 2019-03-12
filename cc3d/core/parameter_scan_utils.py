import shutil
from pathlib import Path
import json
from .CC3DSimulationDataHandler import CC3DSimulationDataHandler

def cc3d_proj_pth_in_output_dir(cc3d_proj_fname: str, output_dir: str) -> Path:
    """

    :param cc3d_proj_fname:
    :param output_dir:
    :return:
    """
    cc3d_proj_pth = Path(cc3d_proj_fname)

    cc3d_proj_dirname = cc3d_proj_pth.parent

    cc3d_proj_dirname_base = cc3d_proj_dirname.parts[-1]

    cc3d_proj_pth_in_out_dir = Path(output_dir).joinpath(cc3d_proj_dirname_base, cc3d_proj_pth.parts[-1])

    return cc3d_proj_pth_in_out_dir



def copy_project_to_output_folder(cc3d_proj_fname: str, output_dir: str) -> bool:
    """
    Copies entire cc3d project to the output dir. Returns flag whether the operation suceeded or not
    :param cc3d_proj_fname:
    :param output_dir:
    :return:
    """

    cc3d_proj_target = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname,output_dir=output_dir)

    if not Path(output_dir).exists():
        shutil.copytree(Path(cc3d_proj_fname).parent, cc3d_proj_target.parent)

        return True

    return True


def parse_param_scan(param_scan_fname: str):
    """

    :param param_scan_fname:
    :return:
    """

    with open(param_scan_fname, 'r') as fin:
        return json.load(fin)


def create_param_scan_status(cc3d_proj_fname: str, output_dir: str):
    """

    :param cc3d_proj_fname:
    :return:
    """
    cc3d_simulation_data_handler = CC3DSimulationDataHandler()
    cc3d_simulation_data_handler.readCC3DFileFormat(cc3d_proj_fname)
    cc3d_sim_data = cc3d_simulation_data_handler.cc3dSimulationData

    if cc3d_sim_data.parameterScanResource is None:
        return

    param_scan_file = cc3d_sim_data.parameterScanResource.path
    param_scan_root_elem = parse_param_scan(param_scan_file)
    param_list_elem = param_scan_root_elem['parameter_list']

    for param_name, param_values in param_list_elem.items():
        # adding current_idx
        param_list_elem[param_name]['current_idx'] = 0

    with open(Path(output_dir).joinpath('param_scan_status.json'), 'w') as fout:

        json.dump(param_scan_root_elem,fout, indent=4)


def update_param_scan_status(cc3d_proj_fname: str):
    """

    :param cc3d_proj_fname:
    :return:
    """
