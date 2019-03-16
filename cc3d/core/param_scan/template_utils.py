from typing import Union
from pathlib import Path
from os.path import *
import shutil
from jinja2 import Environment, FileSystemLoader
from glob import glob


def generate_simulation_files_from_template(cc3d_proj_template: Union[Path, str],
                                            simulation_template_name: Union[str, Path], param_dict: dict) -> None:
    """
    Uses jinja2 templating engine to generate actual simulation files from simulation templates
    ( we use jinja2 templating syntax)

    :param cc3d_proj_template: output directory into which simulation files will be generated
    :param simulation_template_name: full path to current cc3d simulation template - a regular cc3d simulation with
    numbers replaced by template labels
    :param param_dict: {dict} - dictionary of template parameters used to replace template labels with actual parameters
    :return None
    """

    # :return : ({str},{str}) - tuple  where first element is a path to cc3d simulation generated using param_dict.
    # The generated simulation is placed in  simulation_dirname


    # absolute path

    simulation_template_dir = Path(cc3d_proj_template).parent

    candidates = glob(str(simulation_template_dir.joinpath('**/*.py')), recursive=True)

    # simulation_dir_path = dirname(simulation_template_name)
    # simulation_corename = basename(simulation_template_name)

    # # copying simulation dir to "hashed" directory
    # shutil.copytree(src=simulation_dir_path, dst=tmp_simulation_template_dir)

    replacement_candidate_globs = ['*.py', '*xml']
    # simulation_templates_path = join(tmp_simulation_template_dir, 'Simulation')
    # generated_simulation_fname = join(tmp_simulation_template_dir, simulation_corename)

    replacement_candidates = []
    for glob_pattern in replacement_candidate_globs:
        candidates = glob(str(simulation_template_dir.joinpath('**', glob_pattern)), recursive=True)
        replacement_candidates.extend(candidates)

    # j2_env = Environment(loader=FileSystemLoader(simulation_templates_path),
    #                      trim_blocks=True)

    # j2_env = Environment(loader=FileSystemLoader(str(simulation_template_dir)), trim_blocks=True)
    # j2_env = Environment(loader=FileSystemLoader(simulation_template_dir.parts[0]), trim_blocks=True)
    # j2_env = Environment(loader=FileSystemLoader(), trim_blocks=True)

    for replacement_candidate_fname in replacement_candidates:
        replacement_candidate_pth = Path(replacement_candidate_fname)

        j2_env = Environment(loader=FileSystemLoader(str(replacement_candidate_pth.parent)), trim_blocks=True)


        filled_out_template_str = j2_env.get_template(str(replacement_candidate_pth.parts[-1])).render(**param_dict)
        # filled_out_template_str = j2_env.get_template(basename(replacement_candidate_fname)).render(**param_dict)
        print('replacement_candidate_fname=', replacement_candidate_fname)
        print(filled_out_template_str)
        with open(replacement_candidate_fname, 'w') as fout:
            fout.write(filled_out_template_str)

    # return generated_simulation_fname, cc3d_proj_template
