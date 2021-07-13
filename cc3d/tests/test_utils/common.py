import fnmatch
import os
from os.path import *
import psutil
import errno

abs_join = lambda *args: abspath(join(*args))


# Path utilities
# --------------

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

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

def clean_leftover_processes(pid=None):
    if pid is None:
        pid = os.getpid()
    current_process = psutil.Process(pid)
    current_child_processes = current_process.children()

    for proc in current_child_processes:
        print ('process_name = ', proc.name())
        if proc.name().lower().startswith('python'):
            print('KILLING PYTHON PROCESS WITH PID=%s', str(proc.pid))
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                print ('Could not locate {} process pid'.format(proc.name(), proc.pid()))