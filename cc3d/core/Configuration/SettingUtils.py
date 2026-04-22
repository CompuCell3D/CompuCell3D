import os
import shutil
import sys
from typing import Any, Dict, Iterable, Optional
from xml.etree import ElementTree
import base64

from cc3d.core.DefaultSettingsData import *
from cc3d.core.logging import log_py
from cc3d.cpp import CompuCell
from .settingdict import SerializerUtil
from cc3d.core.GraphicsOffScreen.primitives import Color, Point2D, Size2D

try:
    from PyQt5.QtGui import QColor
    from PyQt5.QtCore import QPoint, QSize, QByteArray
except ImportError:
    QColor = None
    QPoint = None
    QSize = None
    QByteArray = None


SETTINGS_XML_ROOT_TAG = 'PlayerSettings'
SETTINGS_XML_VERSION = '1.0'
SIMULATION_SETTINGS_EXCLUDED = {
    'ClosePlayerAfterSimulationDone',
    'DebugOutputPlayer',
    'DemosPath',
    'DisplayConsole',
    'DisplayLatticeData',
    'DisplayMinMaxInfo',
    'DisplayModelEditor',
    'FieldIndex',
    'FfmpegLocation',
    'FloatingWindows',
    'GraphicsWinHeight',
    'GraphicsWinWidth',
    'LastDemoEditTime',
    'LastVersionCheckDate',
    'LogLevel',
    'LogToFile',
    'MainWindowPosition',
    'MainWindowPositionDefault',
    'MainWindowPositionFloating',
    'MainWindowSize',
    'MainWindowSizeDefault',
    'MainWindowSizeFloating',
    'NumberOfRecentSimulations',
    'OutputLocation',
    'PlayerSizes',
    'PlayerSizesDefault',
    'PlayerSizesFloating',
    'PlayerSizesFloatingDefault',
    'ProjectLocation',
    'RecentFile',
    'RecentSimulations',
    'RestartPlayerForNewSimulation',
    'ScreenGeometry',
    'TabIndex',
    'ThemeName',
    'UseInternalConsole',
    'WindowsLayout'
}


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


def simulation_setting_names(settings_object) -> list:
    """
    Returns simulation-relevant settings names, excluding global/UI-only entries.
    """
    return sorted(
        setting_name for setting_name in settings_object.names()
        if setting_name not in SIMULATION_SETTINGS_EXCLUDED
    )


def _value_type_name(value: Any) -> str:
    if value is None:
        return 'none'
    if QColor is not None and isinstance(value, QColor):
        return 'color'
    if QPoint is not None and isinstance(value, QPoint):
        return 'point'
    if QSize is not None and isinstance(value, QSize):
        return 'size'
    if QByteArray is not None and isinstance(value, QByteArray):
        return 'bytearray'
    value_type_name = value.__class__.__name__
    if value_type_name == 'Color':
        return 'color'
    if value_type_name == 'Point2D':
        return 'point'
    if value_type_name == 'Size2D':
        return 'size'
    if value_type_name == 'bool':
        return 'bool'
    if value_type_name in {'str', 'int', 'float', 'complex', 'dict', 'list', 'tuple', 'bytearray'}:
        return value_type_name
    raise RuntimeError(f'Unsupported setting value type: {value_type_name}')


def _scalar_to_xml_text(value: Any, value_type: str) -> str:
    if value_type == 'none':
        return ''
    if value_type == 'bool':
        return '1' if value else '0'
    if value_type == 'color':
        return value.name() if hasattr(value, 'name') else value.to_str_rgb()
    if value_type == 'point':
        if hasattr(value, 'x') and hasattr(value, 'y'):
            return f'{value.x()},{value.y()}'
        return str(value)
    if value_type == 'size':
        if hasattr(value, 'width') and hasattr(value, 'height'):
            return f'{value.width()},{value.height()}'
        return str(value)
    if value_type == 'bytearray':
        if QByteArray is not None and isinstance(value, QByteArray):
            value = bytes(value)
        return base64.b64encode(bytes(value)).decode('ascii')
    return str(value)


def _xml_text_to_scalar(text: Optional[str], value_type: str) -> Any:
    text = '' if text is None else text
    if value_type == 'none':
        return None
    if value_type == 'bool':
        return bool(int(text))
    if value_type == 'int':
        return int(text)
    if value_type == 'float':
        return float(text)
    if value_type == 'complex':
        return complex(text)
    if value_type == 'str':
        return text
    if value_type == 'color':
        return Color.from_str_rgb(text)
    if value_type == 'point':
        x, y = [int(v) for v in text.split(',')]
        return Point2D(x=x, y=y)
    if value_type == 'size':
        width, height = [int(v) for v in text.split(',')]
        return Size2D(width=width, height=height)
    if value_type == 'bytearray':
        return bytearray(base64.b64decode(text.encode('ascii')))
    raise RuntimeError(f'Unsupported setting XML scalar type: {value_type}')


def _serialize_xml_value(parent_el, value: Any) -> None:
    value_type = _value_type_name(value)
    parent_el.set('Type', value_type)

    if value_type == 'dict':
        for key, item_value in value.items():
            child_el = ElementTree.SubElement(parent_el, 'Item')
            child_el.set('Key', _scalar_to_xml_text(key, _value_type_name(key)))
            child_el.set('KeyType', _value_type_name(key))
            _serialize_xml_value(child_el, item_value)
        return

    if value_type in {'list', 'tuple'}:
        for item_value in value:
            child_el = ElementTree.SubElement(parent_el, 'Item')
            _serialize_xml_value(child_el, item_value)
        return

    parent_el.text = _scalar_to_xml_text(value, value_type)


def _deserialize_xml_value(parent_el) -> Any:
    value_type = parent_el.attrib['Type']

    if value_type == 'dict':
        result = {}
        for child_el in parent_el.findall('Item'):
            key = _xml_text_to_scalar(child_el.attrib.get('Key'), child_el.attrib['KeyType'])
            result[key] = _deserialize_xml_value(child_el)
        return result

    if value_type == 'list':
        return [_deserialize_xml_value(child_el) for child_el in parent_el.findall('Item')]

    if value_type == 'tuple':
        return tuple(_deserialize_xml_value(child_el) for child_el in parent_el.findall('Item'))

    return _xml_text_to_scalar(parent_el.text, value_type)


def settings_to_xml_element(settings_dict: Dict[str, Any]) -> ElementTree.Element:
    """
    Serializes settings dictionary to XML element.
    """
    root_el = ElementTree.Element(
        SETTINGS_XML_ROOT_TAG,
        {'Version': SETTINGS_XML_VERSION, 'Scope': 'Simulation'}
    )

    settings_el = ElementTree.SubElement(root_el, 'Settings')
    for setting_name in sorted(settings_dict.keys()):
        setting_el = ElementTree.SubElement(settings_el, 'Setting', {'Name': setting_name})
        _serialize_xml_value(setting_el, settings_dict[setting_name])

    return root_el


def settings_to_xml_string(settings_dict: Dict[str, Any]) -> str:
    return ElementTree.tostring(settings_to_xml_element(settings_dict), encoding='unicode')


def serialize_settings_to_xml(settings_object, xml_file_path: str, setting_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Serializes selected settings from SettingsSQL to XML file.
    Returns the serialized dictionary.
    """
    if setting_names is None:
        setting_names = simulation_setting_names(settings_object)

    settings_dict = {
        setting_name: settings_object.getSetting(setting_name)
        for setting_name in setting_names
    }
    root_el = settings_to_xml_element(settings_dict)
    ElementTree.ElementTree(root_el).write(xml_file_path, encoding='utf-8', xml_declaration=True)
    return settings_dict


def xml_element_to_settings_dict(root_el: ElementTree.Element) -> Dict[str, Any]:
    """
    Deserializes settings XML element generated by settings_to_xml_element.
    """
    settings_el = root_el.find('Settings')
    if settings_el is None:
        raise RuntimeError('Malformed settings XML: missing Settings element')
    setting_els = settings_el.findall('Setting')
    if not setting_els:
        raise RuntimeError('Malformed settings XML: missing serialized Setting elements')

    settings_dict = {}
    for setting_el in setting_els:
        settings_dict[setting_el.attrib['Name']] = _deserialize_xml_value(setting_el)
    return settings_dict


def deserialize_settings_xml(xml_file_path: str) -> Dict[str, Any]:
    """
    Deserializes settings XML file generated by serialize_settings_to_xml.
    """
    root_el = ElementTree.parse(xml_file_path).getroot()
    if root_el.tag != SETTINGS_XML_ROOT_TAG:
        raise RuntimeError(f'Unexpected settings XML root tag: {root_el.tag}')
    return xml_element_to_settings_dict(root_el)


def apply_settings_dict_to_sql(settings_dict: Dict[str, Any], settings_object) -> None:
    """
    Applies settings dictionary to SettingsSQL instance.
    """
    for setting_name, setting_value in settings_dict.items():
        settings_object.setSetting(setting_name, setting_value)


def deserialize_settings_xml_to_sql(xml_file_path: str, settings_object) -> Dict[str, Any]:
    """
    Deserializes XML file and writes the resulting settings into SettingsSQL.
    Returns the deserialized dictionary.
    """
    settings_dict = deserialize_settings_xml(xml_file_path)
    apply_settings_dict_to_sql(settings_dict=settings_dict, settings_object=settings_object)
    return settings_dict


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
