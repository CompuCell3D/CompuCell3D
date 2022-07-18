import sys
import shutil
from subprocess import Popen
from collections import OrderedDict
from pathlib import Path
from typing import List, Union
import json
import numpy as np
import math
from copy import deepcopy
from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationDataHandler
from cc3d.core.filelock import FileLock
from cc3d.core.ParameterScanEnums import SCAN_FINISHED_OR_DIRECTORY_ISSUE
import cc3d.core.param_scan
from .template_utils import generate_simulation_files_from_template


class ParamScanStop(Exception):
    pass


def param_scan_complete_signal(output_dir: Union[str, Path]) -> Path:
    """
    returns file name for the parameter scan complete signal
    :param output_dir: {Path, str} main output directory for the parameter scan run
    :return: {Path}
    """
    return Path(output_dir).joinpath('param_scan_status.complete')


def param_scan_status(output_dir: Union[str, Path]) -> Path:
    """
    returns file name for the parameter scan status
    :param output_dir: {Path, str} main output directory for the parameter scan run
    :return: {Path}
    """
    return Path(output_dir).joinpath('param_scan_status.json')


def handle_param_scan_complete(output_dir: Union[str, Path]) -> None:
    """
    Handles parameter scan complete situation
    :param output_dir: {str, Path}
    :return:
    """

    print('Parameter Scan Complete. If you want to run again, please specify different '
          'output directory')
    try:
        param_scan_complete_signal(output_dir=output_dir).touch(exist_ok=False)
    except FileExistsError:
        pass

    sys.exit(SCAN_FINISHED_OR_DIRECTORY_ISSUE)


def cc3d_proj_pth_in_output_dir(cc3d_proj_fname: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """
    returns path to the .cc3d project in the output directory after the project gets copied
    to the target folder.
    todo - this function might me made independent of parameter scan

    :param cc3d_proj_fname:{Path, str} original path to the .cc3d project
    :param output_dir:{str, Path} path to the outut dir
    :return:{Path} path to the .cc3d file in the output folder (after the entire project gets copied over)
    """

    cc3d_proj_pth = Path(cc3d_proj_fname)

    cc3d_proj_dirname = cc3d_proj_pth.parent

    cc3d_proj_dirname_base = cc3d_proj_dirname.parts[-1]

    cc3d_proj_pth_in_out_dir = Path(output_dir).joinpath(cc3d_proj_dirname_base, cc3d_proj_pth.parts[-1])

    return cc3d_proj_pth_in_out_dir


def copy_project_to_output_folder(cc3d_proj_fname: Union[str, Path], output_dir: Union[str, Path]) -> bool:
    """
    Copies entire cc3d project to the output dir. Returns flag whether the operation succeeded or not
    todo - this function might me made independent of parameter scan
    :param cc3d_proj_fname:
    :param output_dir:
    :return:
    """

    cc3d_proj_target = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname, output_dir=output_dir)

    if not cc3d_proj_target.exists():
        shutil.copytree(Path(cc3d_proj_fname).parent, cc3d_proj_target.parent)

        return True

    return True


def parse_param_scan(param_scan_fname: Union[str, Path]) -> dict:
    """
    Parses json file that includes parameter scan specification
    :param param_scan_fname:
    :return:{dict} root json element
    """

    with open(param_scan_fname, 'r') as fin:
        return json.load(fin)


def param_scan_status_path(output_dir: Union[str, Path]) -> Path:
    """
    Given output dir returns path to param scan status file
    :param output_dir: {Path}
    :return:{Path} path to the parameter scan status json file
    """
    return Path(output_dir).joinpath('param_scan_status.json')


def create_param_scan_status(cc3d_proj_fname: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Creates parameter scan in the otput directory
    :param cc3d_proj_fname:{str} file name of .cc3d project that includes parameter scan
    :param output_dir: {str, Path output folder}
    :return:None
    """

    cc3d_simulation_data_handler = CC3DSimulationDataHandler()
    cc3d_simulation_data_handler.read_cc3_d_file_format(cc3d_proj_fname)
    cc3d_sim_data = cc3d_simulation_data_handler.cc3dSimulationData

    if cc3d_sim_data.parameterScanResource is None:
        return

    param_scan_file = cc3d_sim_data.parameterScanResource.path
    param_scan_root_elem = parse_param_scan(param_scan_file)
    param_list_elem = param_scan_root_elem['parameter_list']

    for i, (param_name, param_values) in enumerate(param_list_elem.items()):

        # adding current_idx
        param_list_elem[param_name]['current_idx'] = 0
        try:
            values = param_list_elem[param_name]['values']
        except KeyError:
            values = []

        # executing code in "code" string
        try:
            code_str = param_list_elem[param_name]['code']
        except KeyError:
            code_str = None

        if code_str:
            values_generated = eval(code_str, {'np': np, 'math': math})
            if isinstance(values_generated, np.ndarray):
                values = values_generated.tolist()
            else:
                values = list(values_generated)

        if not len(values):
            raise RuntimeError('Parameter {} has no values associated with it'.format(param_name))

        param_list_elem[param_name]['values'] = values

    # adding element that will keep track of current iteration
    param_scan_root_elem['current_iteration'] = 0

    param_scan_status_pth = param_scan_status_path(output_dir)

    # we do no create param scan status file if such file exists
    if param_scan_status_pth.exists():
        return

    with open(str(param_scan_status_pth), 'w') as fout:
        json.dump(param_scan_root_elem, fout, indent=4)


def read_parameters_from_param_scan_status_file(output_dir: Union[str, Path]) -> tuple:
    """
    reads current set of parameters from parameter scan file. Parses param_scan_status.json
    :param output_dir:
    :return: tuple of dictionaries - first element is a dict with current parameters,
    second one is a json root dictionary for param_scan_status.json
    """

    param_scan_status_pth = param_scan_status(output_dir=output_dir)

    param_scan_status_root = None
    with open(str(param_scan_status_pth), 'r') as fin:
        param_scan_status_root = json.load(fin)

    param_list_dict = param_scan_status_root['parameter_list']

    ret_dict = {
        'current_iteration': param_scan_status_root['current_iteration'],
    }

    param_dict = OrderedDict(
        [(param_name, param_list_dict[param_name]['values'][param_list_dict[param_name]['current_idx']]) for
         param_name in param_list_dict.keys()]
    )

    ret_dict['parameters'] = param_dict

    return ret_dict, param_scan_status_root


def fetch_next_set_of_scan_parameters(output_dir: Union[str, Path]) -> dict:
    """
    Analyzes param_scan_status.json and picks next set of parameters for parameter scan.
    Writes updated param_scan_status.json to the disk. the new copy of param_scan_status.json
    has updated indices of parameters to indicate which iteration has been just complete
    We assume that we are doing exhaustive parameter scan

    This function performs all operations in an exclusive locking mode which means
    that users can run multiple instances of parameter scan script and the locking
    implemented in this function will provide necessary synchronization

    todo - implement other alternative scan methods

    :param output_dir: {str, Path}
    :return: {dict} a dictionary of {param_label:param_value} type that contains current set of scanned parameters
    Templating engine will use those values to create actual simulation out of provided template
    """

    with FileLock(Path(output_dir).joinpath('param_scan_status.lock')):
        ret_dict, param_scan_status_root = read_parameters_from_param_scan_status_file(output_dir=output_dir)
        update_param_scan_status(param_scan_status_root=param_scan_status_root, output_dir=output_dir)

        return ret_dict


def update_param_scan_status(param_scan_status_root: dict, output_dir: str) -> None:
    """
    given root of json element describing parameter scan (param_scan_status_root) and
    output directory  this function writes updated parameter scan status file to the disk

    :param param_scan_status_root:
    :param output_dir:
    :return:
    """

    advance_param_list(param_scan_status_root=param_scan_status_root)

    param_scan_status_pth = param_scan_status(output_dir=output_dir)
    with open(str(param_scan_status_pth), 'w') as fout:
        json.dump(param_scan_status_root, fout, indent=4)


def advance_param_list(param_scan_status_root: dict) -> dict:
    """
    iterator that returns next set of scanned parameters in the form of dict
    that can be used for an easy replacement in templating engine e.g. Jinja2
    :param param_scan_status_root:{dict} - root of the parameter scan status
    :return:
    """

    param_list_dict = param_scan_status_root['parameter_list']

    curr_list = list(map(lambda key: param_list_dict[key]['current_idx'], param_list_dict.keys()))

    max_list = list(map(lambda key: len(param_list_dict[key]['values']) - 1, param_list_dict.keys()))
    try:
        next_state_list = next(next_cartesian_product_from_state(curr_list=curr_list, max_list=max_list))
    except StopIteration:
        print('advance_param_list: finished fetching parameters')
        raise StopIteration('Parameter scan finished')

    for param_name, current_idx in zip(param_list_dict.keys(), next_state_list):
        param_list_dict[param_name]['current_idx'] = current_idx

    # advancing current iteration
    param_scan_status_root['current_iteration'] = param_scan_status_root['current_iteration'] + 1

    return param_list_dict


#
def next_cartesian_product_from_state(curr_list: List[int], max_list: List[int]) -> List[int]:
    """
    Generator that gives next cartesian product combination from a current state

    :param curr_list: {list of ints} current cartesian product combination
    :param max_list:{list of ints} maximum values a given position may assume
    :return: {list} list of indices - one index per parameter that indicate which values should be used for
    the current combination of scanned parameters
    """
    print('curr_list=', curr_list, 'max_list=', max_list)
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
                # if i == list_len - 1 and carry_over:
                if i == list_len - 1:
                    raise ParamScanStop('next_cartesian_product_from_state: Parameter Scan Finished')

            if not carry_over:
                yield curr_list
                break


def run_single_param_scan_simulation(cc3d_proj_fname: Union[str, Path], current_scan_parameters: dict,
                                     # run_script: Union[str, Path] = '',
                                     gui_flag: bool = False,
                                     output_dir: str = None, arg_list: list = []):
    """
    Given the set of scanned parameters This function creates CC3D project (by applying)
    parameter set to the .cc3d template and the runs such newly created simulation
    Runs single CC3D simulation

    :param cc3d_proj_fname:{str, Path} - path to the "original" cc3d project
    :param current_scan_parameters: {str, path} dictionary with current set of parameters for the simulation
    :param output_dir:{str, Path} output directory
    :return:
    """

    current_iteration = current_scan_parameters['current_iteration']

    # make dir for the current simulation
    scan_iteration_output_dir = Path(output_dir).joinpath('scan_iteration_{}'.format(current_iteration))
    try:
        scan_iteration_output_dir.mkdir(parents=True)
    except FileExistsError:
        raise RuntimeError(f'This particular simulation of parameter scan was already processed. '
                           f'If you want to  proceed make sure to delete folder {str(scan_iteration_output_dir)}')

    # write what parameters will be applied for the current itneration fo the scan
    with open(str(scan_iteration_output_dir.joinpath('current_scan_parameters.json')), 'w') as fout:
        json.dump(obj=current_scan_parameters, fp=fout, indent=4)

    # copy template to the iteration output folder
    copy_project_to_output_folder(cc3d_proj_fname=cc3d_proj_fname, output_dir=scan_iteration_output_dir)

    # replace macros in cc3d template (in XML and py) with actual values to create a valid, runnable cc3d project

    cc3d_proj_template = cc3d_proj_pth_in_output_dir(cc3d_proj_fname=cc3d_proj_fname,
                                                     output_dir=scan_iteration_output_dir)

    print("current_scan_parameters=", current_scan_parameters)
    param_dict = current_scan_parameters['parameters']
    generate_simulation_files_from_template(cc3d_proj_template=cc3d_proj_template, param_dict=param_dict)

    # at this point arg_list may have args from main script
    arg_list_local = deepcopy(arg_list)
    arg_list_local += [
        f'--input={cc3d_proj_template}',
        f'--output-dir={cc3d_proj_template.parent}',
        f'--parameter-scan-iteration={current_scan_parameters["current_iteration"]}'
    ]

    exe_script_list = [f'{sys.executable}', '-m', 'cc3d.run_script']
    if gui_flag:
        arg_list_local += ['--exit-when-done']
        exe_script_list = [f'{sys.executable}', '-m', 'cc3d.player5']

    print('Running simulation with current_scan_parameters=', current_scan_parameters)

    popen_args = exe_script_list + arg_list_local
    print('command=', popen_args)

    cc3d_process = Popen(popen_args)
    out, err = cc3d_process.communicate()
    if out:
        print("standard output of subprocess:")
        print(out)
    if err:
        print("standard error of subprocess:")
        print(err)
    print("returncode of param scan command:")

    print('repeat: Running simulation with current_scan_parameters=', current_scan_parameters)

