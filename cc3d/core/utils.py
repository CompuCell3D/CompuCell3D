import fnmatch
import os
from os.path import *
import errno
from pathlib import Path

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
