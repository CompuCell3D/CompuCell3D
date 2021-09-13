import os
import shutil
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# from CompuCell3D.core.pythonSetupScripts.DefaultSettingsData import *
from cc3d.core.DefaultSettingsData import *


def copy_settings(src_setting_path, dst_setting_dir):
    from cc3d.core.Configuration import copy_settings as f
    return f(src_setting_path, dst_setting_dir)


def synchronizeGlobalAndDefaultSettings(default_settings, global_settings, global_settings_path):
    from cc3d.core.Configuration import synchronize_global_and_default_settings as f
    return f(default_settings, global_settings, global_settings_path)


def load_settings(setting_path):
    """
    loads settings - calls _load_sql_settings
    :param setting_path: {str}
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    return _load_sql_settings(setting_path=setting_path)


def _load_sql_settings(setting_path):
    """
    reads sql setting file from the disk
    :param path: {str} path to SQL settings
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    # from CompuCell3D.player5.Configuration.settingdict import SettingsSQL
    # from .settingdict import SettingsSQL
    from cc3d.player5.Configuration.settingdict import SettingsSQL
    from cc3d.core.Configuration.SettingUtils import check_settings_sanity

    # workaround for Windows leftover DefaultSettingPath.pyc
    from os.path import splitext
    setting_path_no_ext, ext = splitext(setting_path)
    if ext.lower() in ['.xml']:
        setting_path = setting_path_no_ext+'.sqlite'

    settings = SettingsSQL(setting_path)
    if not settings:
        return None, None

    problematic_settings = check_settings_sanity(settings_object=settings)
    if len(problematic_settings):
        print('FOUND THE FOLLOWING PROBLEMATIC SETTINGS: ', problematic_settings)

    return settings, setting_path


def _default_setting_path():
    """
    Returns path to default settings
    :return: {str} path
    """
    dirn = os.path.dirname

    if sys.platform.startswith('darwin'):
        return os.path.abspath(
            os.path.join(dirn(dirn(__file__)), 'Configuration_settings', 'osx', SETTINGS_FILE_NAME))
    else:
        return os.path.abspath(
            os.path.join(dirn(dirn(__file__)), 'Configuration_settings', SETTINGS_FILE_NAME))


def _global_settings_dir():
    """
    returns path to the directory wit global settings
    :return: {str}
    """
    global_setting_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), SETTINGS_FOLDER))

    return global_setting_dir


def _global_setting_path():
    """
    Returns global settings path
    :return: {str}
    """
    global_setting_dir = _global_settings_dir()
    global_setting_path = os.path.abspath(
        os.path.join(global_setting_dir, SETTINGS_FILE_NAME))  # abspath normalizes path

    return global_setting_path


def loadSettings(path):
    """
    Loads settings file
    :param path: {str} absolute path to settings file
    :return: {tuple} (settings object - SettingsSQL, abs path to settings file)
    """
    return _load_sql_settings(path)


def loadGlobalSettings():
    """
    loads global settings
    :return: {tuple} (settings object - SettingsSQL, abs path to settings file)
    """

    global_setting_dir = _global_settings_dir()
    global_setting_path = _global_setting_path()

    # create global settings  directory inside user home directory
    if not os.path.isdir(global_setting_dir):
        try:
            os.makedirs(global_setting_dir)

        except:
            exception_str = 'Configuration: ' \
                            'Could not make directory: {} ' \
                            'to store global settings. ' \
                            'Please make sure that you have ' \
                            'appropriate write permissions'.format(global_setting_dir)
            raise RuntimeError(exception_str)

    if not os.path.isfile(global_setting_path):
        default_setting_path = _default_setting_path()

        copy_settings(default_setting_path, global_setting_dir)

    global_settings, global_settings_path = _load_sql_settings(global_setting_path)
    return global_settings, global_settings_path


def loadDefaultSettings():
    """
    loads default settings
    :return: {tuple} (settings object - SettingsSQL, abs path to settings file)
    """
    return _load_sql_settings(_default_setting_path())
