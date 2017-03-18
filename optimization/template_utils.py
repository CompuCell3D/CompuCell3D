from os.path import *
import shutil
from jinja2 import Environment, FileSystemLoader
from glob import glob



def generate_simulation_files_from_template(simulation_dirname, simulation_template_name, param_dict):
    """
    Uses jinja2 templating engine to generate actual simulation files from simulation templates
    ( we use jinja2 templating syntax)
    :param simulation_dirname: output directory into which simulation files will be generated
    :param simulation_template_name: full path to current cc3d simulation template - a regular cc3d simulation with
    numbers replaced by template labels
    :param param_dict: {dict} - dictionary of template parameters used to replace template labels with actual parameters
    :return : ({str},{str}) - tuple  where first element is a path to cc3d simulation generated using param_dict.
    The generated simulation is placed in  simulation_dirname
    """
    # absolute path

    tmp_simulation_template_dir = simulation_dirname


    simulation_dir_path = dirname(simulation_template_name)
    simulation_corename = basename(simulation_template_name)

    # copying simulation dir to "hashed" directory
    shutil.copytree(src=simulation_dir_path, dst=tmp_simulation_template_dir)

    replacement_candidate_globs = ['*.py', '*xml']
    simulation_templates_path = join(tmp_simulation_template_dir, 'Simulation')
    generated_simulation_fname = join(tmp_simulation_template_dir, simulation_corename)

    replacement_candidates = []
    for glob_pattern in replacement_candidate_globs:
        replacement_candidates.extend(glob(simulation_templates_path + '/' + glob_pattern))

    j2_env = Environment(loader=FileSystemLoader(simulation_templates_path),
                         trim_blocks=True)

    for replacement_candidate_fname in replacement_candidates:
        filled_out_template_str = j2_env.get_template(basename(replacement_candidate_fname)).render(**param_dict)
        with open(replacement_candidate_fname, 'w') as fout:
            fout.write(filled_out_template_str)

    return generated_simulation_fname, simulation_dirname