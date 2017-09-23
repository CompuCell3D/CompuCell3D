from os.path import *
from DefaultSettingsData import *
from settingdict import SettingDict

defaultSettings, defaultSettingsPath = loadDefaultSettings()



def loadDefaultSettings():
    default_setting_path = abspath(join(dirname(__file__), SETTINGS_FILE_NAME_DB))  # abspath normalizes path

    defaultSettings = loadSettings(default_setting_path)

    if not defaultSettings:
        return None, None

    return defaultSettings, default_setting_path

def loadSettings(setting_path):

    setting_dict = SettingDict(setting_path)
    return setting_dict