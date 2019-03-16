from typing import Union
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from glob import glob


def generate_simulation_files_from_template(cc3d_proj_template: Union[Path, str], param_dict: dict) -> None:
    """
    Uses jinja2 templating engine to generate actual simulation files from simulation templates
    Important: This function overwrites all the files that have jinja2 macros and therefore
    cc3d_proj_template should point a .cc3d file that can be overwritten. Typically we make a copy of the
    original parameter scan template and overwrite a copy.
    ( we use jinja2 templating syntax)

    :param cc3d_proj_template: {str,PAth} path to the .cc3d project template that is to be overwritten
    IMPORTANT: files located inside .cc3d project pointed by cc3d_proj_template will be overwritten

    :param param_dict: {dict} - dictionary of template parameters used to replace template labels with actual parameters
    :return None
    """
    simulation_template_dir = Path(cc3d_proj_template).parent

    replacement_candidate_globs = ['*.py', '*xml']

    replacement_candidates = []
    for glob_pattern in replacement_candidate_globs:
        candidates = glob(str(simulation_template_dir.joinpath('**', glob_pattern)), recursive=True)
        replacement_candidates.extend(candidates)

    for replacement_candidate_fname in replacement_candidates:
        replacement_candidate_pth = Path(replacement_candidate_fname)

        j2_env = Environment(loader=FileSystemLoader(str(replacement_candidate_pth.parent)), trim_blocks=True)

        filled_out_template_str = j2_env.get_template(str(replacement_candidate_pth.parts[-1])).render(**param_dict)

        with open(replacement_candidate_fname, 'w') as fout:
            fout.write(filled_out_template_str)
