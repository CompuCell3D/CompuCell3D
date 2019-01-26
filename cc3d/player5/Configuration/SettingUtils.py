import os
import shutil
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# from CompuCell3D.core.pythonSetupScripts.DefaultSettingsData import *
from cc3d.core.DefaultSettingsData import *


def load_settings(setting_path):
    """
    loads settings - calls _load_sql_settings
    :param setting_path: {str}
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    return _load_sql_settings(setting_path=setting_path)

def _check_settings_sanity(settings_object):
    """
    Checks whether all settings listed in settings_object are accessible.
    :param settings_object: {instance of SettingsSQL} settings object
    :return: {list of str} list of settings that are problematic
    """
    problematic_settings = []
    for setting_name in settings_object.names():
        try:
            settings_object.getSetting(setting_name)
        except:
            problematic_settings.append(setting_name)
    return problematic_settings

def _load_sql_settings(setting_path):
    """
    reads sql setting file from the disk
    :param path: {str} path to SQL settings
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    # from CompuCell3D.player5.Configuration.settingdict import SettingsSQL
    # from .settingdict import SettingsSQL
    from cc3d.player5.Configuration.settingdict import SettingsSQL

    # workaround for Windows leftover DefaultSettingPath.pyc
    from os.path import splitext
    setting_path_no_ext, ext = splitext(setting_path)
    if ext.lower() in ['.xml']:
        setting_path = setting_path_no_ext+'.sqlite'

    settings = SettingsSQL(setting_path)
    if not settings:
        return None, None

    problematic_settings = _check_settings_sanity(settings_object=settings)
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


def copy_settings(src_setting_path, dst_setting_dir):
    """
    Copies settings file specified by src_setting_path to directory specified by dst_setting_dir
    :param src_setting_path: {str} full path to settins file
    :param dst_setting_dir: {str} full path to targe directory for settings
    :return: None
    """

    try:
        shutil.copy(src_setting_path, dst_setting_dir)
    except:
        exception_str = 'Configuration: ' \
                        'Could not copy setting file: {} ' \
                        'to {} directory. ' \
                        'Please make sure that you have ' \
                        'appropriate write permissions'.format(src_setting_path, dst_setting_dir)
        raise RuntimeError(exception_str)


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


def synchronizeGlobalAndDefaultSettings(default_settings, global_settings, global_settings_path):
    """
    Synchronizes global settings and default settings. This function checks for
    new settings in the default settings file
    :param default_settings: {SettingsSQL} settings object with default settings
    :param global_settings: {SettingsSQL} settings object with global settings
    :param global_settings_path: {str} path to global settings file
    :return: None
    """

    default_settings_names = default_settings.names()
    global_settings_names = global_settings.names()

    new_setting_names = set(default_settings_names) - set(global_settings_names)

    for new_setting_name in new_setting_names:
        print('new_setting_name = ', new_setting_name)
        new_default_setting_val = default_settings.setting(new_setting_name)

        global_settings.setSetting(new_setting_name, new_default_setting_val)
