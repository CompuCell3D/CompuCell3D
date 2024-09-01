import os
import shutil
import sys
from typing import Any, Dict
from xml.etree import ElementTree

from cc3d.core.DefaultSettingsData import *
from cc3d.core.logging import log_py
from cc3d.cpp import CompuCell
from .settingdict import SerializerUtil


def load_settings(setting_path):
    """
    loads settings - calls _load_sql_settings

    :param setting_path: {str}
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    return _load_sql_settings(setting_path=setting_path)


def loadSettings(path):
    """
    Loads settings file

    :param path: {str} absolute path to settings file
    :return: {tuple} (settings object - SettingsSQL, abs path to settings file)
    """
    return load_settings(path)


def _check_settings_sanity(settings_object):
    problematic_settings = []
    for setting_name in settings_object.names():
        try:
            settings_object.getSetting(setting_name)
        except:
            problematic_settings.append(setting_name)
    return problematic_settings


def check_settings_sanity(settings_object):
    """
    Checks whether all settings listed in settings_object are accessible.

    :param settings_object: {instance of SettingsSQL} settings object
    :return: {list of str} list of settings that are problematic
    """
    return _check_settings_sanity(settings_object)


def _load_sql_settings(setting_path):
    """
    reads sql setting file from the disk

    :param path: {str} path to SQL settings
    :return: {tuple} (SettingsSQL object, path to SQL settings)
    """
    from cc3d.core.Configuration.settingdict import SettingsSQL

    # workaround for Windows leftover DefaultSettingPath.pyc
    from os.path import splitext, isdir, isfile, dirname
    setting_path_no_ext, ext = splitext(setting_path)
    if ext.lower() in ['.xml']:
        setting_path = setting_path_no_ext+'.sqlite'

    # Verify existing settings file or create if necessary
    if not isdir(dirname(setting_path)):
        os.makedirs(dirname(setting_path), exist_ok=True)
    if not isfile(setting_path):
        with open(setting_path, 'w'):
            pass

    settings = SettingsSQL(setting_path)
    if not settings:
        return None, None

    problematic_settings = _check_settings_sanity(settings_object=settings)
    if len(problematic_settings):
        print('FOUND THE FOLLOWING PROBLEMATIC SETTINGS: ', problematic_settings)

    return settings, setting_path


def _default_setting_dir() -> str:
    """
    Returns path to default settings directory
    """
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Configuration_settings')


def _default_setting_path():
    """
    Returns path to default settings

    :return: {str} path
    """
    if sys.platform.startswith('darwin'):
        return os.path.abspath(
            os.path.join(_default_setting_dir(), 'osx', SETTINGS_FILE_NAME))
    else:
        return os.path.abspath(
            os.path.join(_default_setting_dir(), SETTINGS_FILE_NAME))


def _default_setting_path_xml() -> str:
    """
    Returns the path to the default settings xml data
    """
    if sys.platform.startswith('darwin'):
        return os.path.join(_default_setting_dir(), SETTINGS_XML_OSX_FILE_NAME)
    else:
        return os.path.join(_default_setting_dir(), SETTINGS_XML_FILE_NAME)


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


def load_global_settings():
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


def loadGlobalSettings():
    return load_global_settings()


def load_default_settings():
    """
    loads default settings

    :return: {tuple} (settings object - SettingsSQL, abs path to settings file)
    """
    return _load_sql_settings(_default_setting_path())


def loadDefaultSettings():
    return load_default_settings()


def synchronize_global_and_default_settings(default_settings, global_settings, global_settings_path):
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


def synchronizeGlobalAndDefaultSettings(default_settings, global_settings, global_settings_path):
    return synchronize_global_and_default_settings(default_settings, global_settings, global_settings_path)


def _handle_settings_value_xml(_val: str):
    """
    Handles data intended for sqlite but used in python types
    """
    if _val == 'True':
        return '1'
    elif _val == 'False':
        return '0'
    return _val


def _warn_setting_not_set(_el):
    log_py(CompuCell.LOG_WARNING, f'Setting not set: {_el.attrib["Name"]}')


def default_settings_dict_xml() -> Dict[str, Any]:
    """
    Returns default data dictionary
    """
    fp = _default_setting_path_xml()
    if not os.path.isfile(fp):
        log_py(CompuCell.LOG_WARNING, f'Could not located default settings data ({fp})')
        return {}

    data_root = ElementTree.parse(fp).getroot()
    sz_util = SerializerUtil()

    def _import_data(_el):
        _val_text = _handle_settings_value_xml(_el.text)
        _type_fcn = sz_util.type_2_deserializer_dict[_el.attrib['Type']]
        return _el.attrib['Name'], _type_fcn(_val_text)

    # "e" is the XML tag of interest for settings
    result = {}
    for el in data_root.findall('Settings'):
        for el_e in el.findall('e'):
            if el_e.attrib['Type'] == 'dict':
                result_e = {}
                for el_e_e in el_e.findall('e'):
                    try:
                        result_e.__setitem__(*_import_data(el_e_e))
                    except (KeyError, TypeError):
                        _warn_setting_not_set(el_e_e)
                result[el_e.attrib['Name']] = result_e
            else:
                try:
                    result.__setitem__(*_import_data(el_e))
                except (KeyError, TypeError):
                    _warn_setting_not_set(el_e)

    return result
