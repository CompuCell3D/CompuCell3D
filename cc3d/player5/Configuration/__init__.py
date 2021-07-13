###### NOTE
# DefaultSettings.py defines location of the global setting file
# SettingUtils.py provides most functionality as far as Setting lookup, loading writing etc...


###### NOTE 2
# To add new setting edit glolbal _settings.xml and create placeholder for the new setting
# e.g.  
# add
# <e Name="WindowsLayout" Type="dict">
# </e>

# Now this setting is treated as "known setting " and you can manipulat it using Configuration set/get setting fuctions/. For more complex settings you may need to write 
# some auxiliary functions facilitating translation from Setting format to Python format - this usually applies to e.g. dictionaries of dictionaries

# todo - handle bad file format for settings
# todo - at the beginning read all settings and make sure there are no issues in the stored settings
# todo - see if updateFieldsParams() function is needed
# todo - see if we need syncPreferences

import os
from .SettingUtils import *
from .SettingUtils import _global_setting_path

# the imports have to be fixed in the entire CC3D!!!!
# try:
#     from CompuCell3D.player5.Configuration.settingdict import SettingsSQL
# except ImportError:
#     from Config.settingdict import SettingsSQL
from .settingdict import SettingsSQL

LATTICE_TYPES = {"Square": 1, "Hexagonal": 2}

maxNumberOfRecentFiles = 5

MODULENAME = '------- player5/Configuration/__init__.py: '


class Configuration:
    """
    "Global" class that serves as a container for global, default and simulation specific settings and their
    respective paths
    """
    defaultSettings, defaultSettingsPath = loadDefaultSettings()
    myGlobalSettings, myGlobalSettingsPath = loadGlobalSettings()

    synchronizeGlobalAndDefaultSettings(defaultSettings, myGlobalSettings, myGlobalSettingsPath)

    # stores simulation-specific settings for  i.e. ones which are associated with individual cc3d project
    myCustomSettings = None
    myCustomSettingsPath = ''

    globalOnlySettings = ['RecentSimulations', 'NumberOfRecentSimulations', 'OutputLocation', 'ProjectLocation',
                          'FloatingWindows', 'MainWindowSizeDefault', 'MainWindowSizeDefault', 'ScreenGeometry']
    customOnlySettings = ['WindowsLayout', 'Types3DInvisible']

    activeFieldNamesList = []

    # dictionary of FieldParams Settings
    simFieldsParams = None


def initConfiguration():
    """
    Function that "flushes" al configuration settings . Called when user hits "Stop" in the player
    :return:
    """
    Configuration.defaultSettings, Configuration.defaultSettingsPath = loadDefaultSettings()
    Configuration.myGlobalSettings, Configuration.myGlobalSettingsPath = loadGlobalSettings()
    synchronizeGlobalAndDefaultSettings(Configuration.defaultSettings, Configuration.myGlobalSettings,
                                        Configuration.myGlobalSettingsPath)
    Configuration.myCustomSettings = None
    Configuration.myCustomSettingsPath = ''


def replace_custom_settings_with_defaults():
    """
    Replaces simulation-specific settings with the default settings
    :return: None
    """
    default_settings, default_settings_path = loadDefaultSettings()

    copy_settings(src_setting_path=default_settings_path,
                  dst_setting_dir=os.path.dirname(Configuration.myCustomSettingsPath))

    Configuration.myCustomSettings, Configuration.myCustomSettingsPath = loadSettings(
        Configuration.myCustomSettingsPath)


def restore_default_global_settings():
    """
    Removes global settings
    :return:
    """

    default_settings, default_settings_path = loadDefaultSettings()
    copy_settings(src_setting_path=default_settings_path,
                  dst_setting_dir=os.path.dirname(Configuration.myGlobalSettingsPath))

    Configuration.myGlobalSettings, Configuration.myGlobalSettingsPath = loadSettings(
        Configuration.myGlobalSettingsPath)

def getSettingNameList():
    """
    Returns a list of setting names (those stored in the global setting file).
    :return: {list os str}
    """
    return Configuration.myGlobalSettings.names()


def setUsedFieldNames(fieldNamesList):
    """
    Function that examines 'FieldParams' settings and keeps only those that are
    associated with field names listed in the 'fieldNamesList'. Settings for fields
    not listed in 'fieldNamesList' are discarded
    :param fieldNamesList: {list of str} list of fields whose settings will be remain in the settings file
    :return: None
    """
    Configuration.activeFieldNamesList = fieldNamesList
    fieldParams = getSetting('FieldParams')

    # purging uneeded fields
    cleanedFieldParams = {}
    for fieldName in Configuration.activeFieldNamesList:
        try:

            cleanedFieldParams[fieldName] = fieldParams[fieldName]

        except KeyError:
            cleanedFieldParams[fieldName] = getDefaultFieldParams()
            # cleanedFieldParams[fieldName] = 

            pass

    setSetting('FieldParams', cleanedFieldParams)


def writeAllSettings():
    """
    Kept to satisfy legacy API - not needed with sql-based settings
    :return: None
    """
    pass


def writeSettingsForSingleSimulation(path):
    """
    Here we are creating settings for a single simulation or loading them if they already exist
    :param path: {src} abs path to local settings
    :return: None
    """
    if not os.path.isfile(path):
        copy_settings(src_setting_path=_global_setting_path(), dst_setting_dir=os.path.dirname(path))
        Configuration.myCustomSettings, Configuration.myCustomSettingsPath = load_settings(path)

    else:
        Configuration.myCustomSettings, Configuration.myCustomSettingsPath = load_settings(path)


def initializeCustomSettings(filename):
    """
    Loads simulation-specific settings
    :param filename: {str} absolute path to the simulation-specific setting file
    :return: None
    """
    Configuration.myCustomSettings, Configuration.myCustomSettingsPath = loadSettings(filename)


def getDefaultFieldParams():
    """
    Creates dictionary for FieldParams setting (specifying how to diplay given field).
    All field params have default values
    :return: {dict} Dictionary of field parameters (field visualzation parameters)
    """

    paramsDict = {}

    paramsDict["MinRange"] = getSetting("MinRange")
    paramsDict["MinRangeFixed"] = getSetting("MinRangeFixed")
    paramsDict["MaxRange"] = getSetting("MaxRange")
    paramsDict["MaxRangeFixed"] = getSetting("MaxRangeFixed")

    paramsDict["ShowPlotAxes"] = getSetting("ShowPlotAxes")

    paramsDict["NumberOfLegendBoxes"] = getSetting("NumberOfLegendBoxes")
    paramsDict["NumberAccuracy"] = getSetting("NumberAccuracy")
    paramsDict["LegendEnable"] = getSetting("LegendEnable")

    paramsDict["NumberOfContourLines"] = getSetting("NumberOfContourLines")

    paramsDict["ScaleArrowsOn"] = getSetting("ScaleArrowsOn")
    color = getSetting("ArrowColor")

    paramsDict["ArrowColor"] = color

    paramsDict["ArrowLength"] = getSetting("ArrowLength")
    paramsDict["FixedArrowColorOn"] = getSetting("FixedArrowColorOn")
    paramsDict["OverlayVectorsOn"] = getSetting("OverlayVectorsOn")
    paramsDict["ScalarIsoValues"] = getSetting("ScalarIsoValues")
    paramsDict["ContoursOn"] = getSetting("ContoursOn")

    return paramsDict


# def initFieldsParams(fieldNames):
#     """
#     called from SimpleTabView once we know the fields
#     :param fieldNames:
#     :return:
#     """
#
#
#
#     fieldParams = getSetting('FieldParams')
#
#     for field in fieldNames:
#
#         if field not in fieldParams.keys() and field != 'Cell_Field':
#             fieldParams[field] = getDefaultFieldParams()
#
#     Configuration.simFieldsParams = fieldParams
#     # print 'initFieldsParams fieldParams = ',fieldParams
#     setSetting('FieldParams', fieldParams)


def updateFieldsParams(fieldName, fieldDict):
    # todo - see if this function is needed
    """
    Called by ConfigurationDialog - stores field params dictionary (fieldDict) associated with field (fieldName)
    in the Configuration.simFieldsParams. Also stores this information in the settings file on the hard drive

    NOT SURE IF THIS FUNCTION IS ACTUALLY NEEDED
    :param fieldName: {str} name of the field
    :param fieldDict: {dict} dictionary of field-related visualization settings
    :return: None
    """
    fieldParamsDict = getSetting("FieldParams")
    Configuration.simFieldsParams = fieldParamsDict

    fieldName = str(fieldName)

    print('CONFIGURATION: fieldName =', fieldName)
    print('CONFIGURATION: fieldDict =', fieldDict)

    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    setSetting('FieldParams', Configuration.simFieldsParams)

    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not


def getRecentSimulationsIncludingNewItem(simName):
    """
    Returns a list of recent simulations. Makes sure that the list does not exceed
    number of items specified in the 'NumberOfRecentSimulations' setting
    :param simName: {str}
    :return: {list of str} list of recent simulations
    """

    tmp_list = getSetting('RecentSimulations')
    max_length = getSetting('NumberOfRecentSimulations')

    current_strlist = [x for x in tmp_list]

    # inserting new element
    current_strlist.insert(0, simName)

    # eliminating duplicates
    seen = set()
    seen_add = seen.add
    current_strlist = [x for x in current_strlist if not (x in seen or seen_add(x))]

    # ensuring that we keep only NumberOfRecentSimulations elements
    if len(current_strlist) > max_length:
        current_strlist = current_strlist[: - (len(current_strlist) - max_length)]

    return current_strlist


# we append an optional fieldName now to allow for field-dependent parameters from Prefs
def getSetting(_key, fieldName=None):
    """
    Retrieves setting name from the setting file(s)/database(s)
    :param _key: {str} name of the setting
    :param fieldName:{str} name of the field - optional and used only in conjunction with dictionary of
    settings associated with field visualization parameters
    :return: {object} object corresponding to a given setting
    """

    if Configuration.myCustomSettings:
        setting_storage = Configuration.myCustomSettings

    else:
        setting_storage = Configuration.myGlobalSettings

    # some settings are stored in the global settings e.g. number of recent simulations or recent simulations list
    if _key in Configuration.globalOnlySettings:
        setting_storage = Configuration.myGlobalSettings

    if fieldName is not None:
        field_params = getSetting('FieldParams')
        try:
            single_field_params = field_params[fieldName]

            return single_field_params[_key]
        except LookupError:
            pass  # returning global parameter for the field

    # val = settingStorage.getSetting(_key)
    # a way to fetch unknown setting from default setting and writing it back to the custom settins
    try:
        val = setting_storage.getSetting(_key)
    except KeyError:
        # attempt to fetch setting from global setting
        setting_storage = Configuration.myGlobalSettings
        val = setting_storage.getSetting(_key)
        # and write it to custom setting
        if Configuration.myCustomSettings:
            setting_storage = Configuration.myCustomSettings
            setting_storage.setSetting(_key, val)

    return val


def setSetting(_key, _value):
    """
    stores object (_value) under name (_key) in the relevant setting file(s)/databases(s)
    :param _key: {str} name of the setting
    :param _value: {object} object associated with the setting
    :return: None
    """

    val = _value

    if _key == 'RecentSimulations':  # need to find better solution for that...
        simName = _value
        val = getRecentSimulationsIncludingNewItem(simName)  # val is the value that is stored int hte settings

    # some settings are stored in the global settings e.g. number of recent simulations or recent simulations list
    if _key in Configuration.globalOnlySettings:
        Configuration.myGlobalSettings.setSetting(_key, val)

    elif _key in Configuration.customOnlySettings:  # some settings are stored in the custom settings e.g. WindowsLayout

        if Configuration.myCustomSettings:
            Configuration.myCustomSettings.setSetting(_key, val)

    else:
        Configuration.myGlobalSettings.setSetting(_key, val)

        if Configuration.myCustomSettings:
            Configuration.myCustomSettings.setSetting(_key, val)


def syncPreferences():
    """
    this function invoked when we close the Prefs dialog with the "OK" button.
    Not used with sql-based settings. Kept to satisfy legacy API
    :return:
    """

    pass
