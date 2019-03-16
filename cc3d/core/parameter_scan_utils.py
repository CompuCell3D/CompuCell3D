import time
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List, Union
import json
from .CC3DSimulationDataHandler import CC3DSimulationDataHandler
from cc3d.core.filelock import FileLock


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

    cc3d_proj_target = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

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

def param_scan_status_path(output_dir:Union[str,Path])->Path:
    """
    given output dir returns path to param scan status file
    :param output_dir: {Path}
    :return:
    """
    return Path(output_dir).joinpath('param_scan_status.json')


def create_param_scan_status(cc3d_proj_fname: str, output_dir: str)->None:
    """
    Creates parameter scan in the otput directory
    :param cc3d_proj_fname:{str} file name ofcc3d project that includes parameter scan
    :return:None
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


    param_scan_status_pth = param_scan_status_path(output_dir)

    # we do no create param scan status fil iof such file exists
    if param_scan_status_pth.exists():
        return

    with open(str(param_scan_status_pth), 'w') as fout:

        json.dump(param_scan_root_elem, fout, indent=4)

def fetch_next_set_of_scan_parameters(output_dir):
    # todo add locking
    with FileLock(Path(output_dir).joinpath('param_scan_status.lock')):
        time.sleep(3.0)
        param_scan_status_path = Path(output_dir).joinpath('param_scan_status.json')

        with open(param_scan_status_path, 'r') as fin:
            param_scan_status_root = json.load(fin)

        param_list_dict = param_scan_status_root['parameter_list']

        ret_dict = OrderedDict(
            [ (param_name,param_list_dict[param_name]['values'][param_list_dict[param_name]['current_idx']]) for param_name in param_list_dict.keys()]
        )

        update_param_scan_status(param_scan_status_root=param_scan_status_root, output_dir=output_dir)

        return ret_dict

def update_param_scan_status(param_scan_status_root:dict, output_dir:str)->dict:
    """

    :param param_list_dict:
    :return:
    """
    param_list_dict = param_scan_status_root['parameter_list']

    param_list_dict = advance_param_list(param_list_dict=param_list_dict)

    param_scan_status_path = Path(output_dir).joinpath('param_scan_status.json')
    with open(param_scan_status_path,'w') as fout:
        json.dump(param_scan_status_root,fout, indent=4)




def advance_param_list(param_list_dict: dict) -> dict:
    """
    iterator that returns next set of scanned parameters in the form of dict
    that can be used for an easy replacement in templating engine e.g. Jinja2
    :param output_dir:
    :return:
    """
    # # todo add locking
    # param_scan_status_path = Path(output_dir).joinpath('param_scan_status.json')
    #
    # with open(param_scan_status_path, 'r') as fin:
    #     param_scan_status_root = json.load(fin)
    #
    # param_list_dict = param_scan_status_root['parameter_list']

    curr_list = list(map(lambda key: param_list_dict[key]['current_idx'],  param_list_dict.keys() ))
    max_list = list(map(lambda key: len(param_list_dict[key]['values'])-1,  param_list_dict.keys() ))

    next_state_list = next(next_cartesian_product_from_state(curr_list=curr_list,max_list=max_list))

    for param_name, current_idx in zip(param_list_dict.keys(), next_state_list):
        param_list_dict[param_name]['current_idx'] = current_idx

    return param_list_dict
    # ret_dict = OrderedDict(
    #     [ (param_name,param_list_dict[param_name]['values'][value_idx]) for param_name, value_idx in zip(param_list_dict.keys(), next_state_list)]
    # )
    #
    # # ret_dict = dict(zip(param_list_dict.keys(), next_state_list))
    #
    # return ret_dict

# def update_param_scan_status(cc3d_proj_fname: str):
#     """
#
#     :param cc3d_proj_fname:
#     :return:
#     """


def next_cartesian_product_from_state(curr_list: List[int], max_list: List[int]) -> List[int]:
    """
    a generator that gives next cartesian product combination from a current state
    :param curr_list: {list of ints} current cartesian product combination
    :param max_list:{list of ints} maximum values a given position may assume
    :return:
    """

    if len(curr_list) != len(max_list):
        raise ValueError("curr_list and max_list must have same length ")

    list_len = len(curr_list)

    while True:
        for i in range(len(curr_list)):
            curr_list[i] += 1
            carry_over = 0
            if curr_list[i] > max_list[i]:
                curr_list[i] = 0
                carry_over = 1
                if i == list_len - 1:
                    raise StopIteration()

            if not carry_over:
                yield curr_list
                break

def run_single_param_scan_simulation(output_dir:str=None):
    """

    :param output_dir:
    :return:
    """

    time.sleep(1.0)