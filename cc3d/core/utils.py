import fnmatch
import os
from os.path import *
import errno
from pathlib import Path
import subprocess
import sys

abs_join = lambda *args: abspath(join(*args))
# Path utilities
# --------------

def mkdir_p(path):

    Path(path).mkdir(parents=True, exist_ok=True)
    # try:
    #     os.makedirs(path)
    # except OSError as exc:
    #     if exc.errno == errno.EEXIST and os.path.isdir(path):
    #         pass
    #     else:
    #         raise

def find_file_in_dir(dirname, fname_pattern):
    """
    Does recursive filename match in the directory. Returns all matches found in the directory
    :param dirname: directory to search
    :param fname_pattern: file name pattern (wildcard expression)
    :return: list of matches
    """

    matches = []
    for root, dirnames, filenames in os.walk(dirname):
        for filename in fnmatch.filter(filenames, fname_pattern):
            matches.append(os.path.join(root, filename))
    return matches


def find_current_conda_env(conda_exec):
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    if env_name:
        return env_name

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        return Path(conda_prefix).name

    if conda_exec is None:
        return None

    envs = subprocess.check_output(f'{conda_exec} env list', shell=True).splitlines()
    active_envs = []
    sys_prefix = Path(sys.prefix).resolve()
    for env_line in envs:
        decoded_env_line = env_line.decode("utf-8").strip()
        if not decoded_env_line or decoded_env_line.startswith('#'):
            continue
        if '*' in decoded_env_line:
            active_envs.append(decoded_env_line)
            continue

        env_line_parts = decoded_env_line.split()
        env_path = Path(env_line_parts[-1])
        try:
            if env_path.resolve() == sys_prefix:
                return env_line_parts[0]
        except OSError:
            pass

    if not active_envs:
        return None

    env_name = active_envs[0].split()[0]
    return env_name


def find_conda():
    conda_exec = None
    python_exec = Path(sys.executable)
    python_exec_dir = python_exec.parent
    print('python_exec_dir=', python_exec_dir)

    if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):

        conda_exec_candidates = [
            # if using other env
            Path().joinpath(*python_exec_dir.parts[:-3]).joinpath('bin', 'conda'),
            # if using python.app or similar from env
            Path().joinpath(*python_exec_dir.parts[:-5]).joinpath('bin', 'conda'),
            # if using base conda env
            python_exec_dir.joinpath('conda'),
        ]

        for candidate in conda_exec_candidates:
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                conda_exec = candidate
                break

        print('conda_exec=', conda_exec)
        os.system(str(conda_exec))
    elif sys.platform.startswith('win'):

        conda_exec_candidates = [
            # if using other env
            Path().joinpath(*python_exec_dir.parts[:-2]).joinpath('condabin', 'conda.bat'),
            # if using python.app or similar from env

            # if using base conda env
            python_exec_dir.joinpath('condabin', 'conda.bat'),
        ]

        for candidate in conda_exec_candidates:
            if candidate.exists():
                conda_exec = candidate
                break

    return conda_exec
