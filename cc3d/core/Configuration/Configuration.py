import os
from .SettingUtils import copy_settings, load_settings, _global_setting_path
from .SettingUtils import load_global_settings, load_default_settings, synchronize_global_and_default_settings


class Configuration:
    """
    Serves as a container for global, default and simulation specific settings and their respective paths
    """

    defaultSettings, defaultSettingsPath = load_default_settings()

    def __init__(self):
        self.globalOnlySettings = ['OutputLocation', 'ProjectLocation']
        self.myGlobalSettings, self.myGlobalSettingsPath = load_global_settings()

        synchronize_global_and_default_settings(self.defaultSettings, self.myGlobalSettings, self.myGlobalSettingsPath)

        # stores simulation-specific settings for  i.e. ones which are associated with individual cc3d project
        self.myCustomSettings = None
        self.myCustomSettingsPath = ''
        self.customOnlySettings = []

        self.activeFieldNamesList = []

        # dictionary of FieldParams Settings
        self.simFieldsParams = None

    def replace_custom_settings_with_defaults(self):
        """
        Replaces simulation-specific settings with the default settings

        :return: None
        """
        default_settings, default_settings_path = load_default_settings()

        copy_settings(src_setting_path=default_settings_path,
                      dst_setting_dir=os.path.dirname(self.myCustomSettingsPath))

        self.myCustomSettings, self.myCustomSettingsPath = load_settings(self.myCustomSettingsPath)

    def restore_default_global_settings(self):
        """
        Removes global settings

        :return:
        """
        default_settings, default_settings_path = load_default_settings()
        copy_settings(src_setting_path=default_settings_path,
                      dst_setting_dir=os.path.dirname(self.myGlobalSettingsPath))

        self.myGlobalSettings, self.myGlobalSettingsPath = load_settings(self.myGlobalSettingsPath)

    def get_setting_name_list(self):
        """
        Returns a list of setting names (those stored in the global setting file).

        :return: {list os str}
        """
        return self.myGlobalSettings.names()

    def getSettingNameList(self):
        """
        Returns a list of setting names (those stored in the global setting file).

        :return: {list os str}
        """
        return self.get_setting_name_list()

    def set_used_field_names(self, field_names_list):
        """
        Function that examines 'FieldParams' settings and keeps only those that are
        associated with field names listed in the 'fieldNamesList'. Settings for fields
        not listed in 'fieldNamesList' are discarded

        :param field_names_list: {list of str} list of fields whose settings will be remain in the settings file
        :return: None
        """
        self.activeFieldNamesList = field_names_list
        fieldParams = self.get_setting('FieldParams')

        # purging uneeded fields
        cleanedFieldParams = {}
        for fieldName in self.activeFieldNamesList:
            try:
                cleanedFieldParams[fieldName] = fieldParams[fieldName]
            except KeyError:
                cleanedFieldParams[fieldName] = self.get_default_field_params()
                pass

        self.set_setting('FieldParams', cleanedFieldParams)

    def setUsedFieldNames(self, fieldNamesList):
        return self.set_used_field_names(fieldNamesList)

    def writeAllSettings(self):
        """
        Kept to satisfy legacy API - not needed with sql-based settings

        :return: None
        """
        pass

    def write_settings_for_single_simulation(self, path):
        """
        Here we are creating settings for a single simulation or loading them if they already exist

        :param path: {src} abs path to local settings
        :return: None
        """
        if not os.path.isfile(path):
            copy_settings(src_setting_path=_global_setting_path(), dst_setting_dir=os.path.dirname(path))
            self.myCustomSettings, self.myCustomSettingsPath = load_settings(path)

        else:
            self.myCustomSettings, self.myCustomSettingsPath = load_settings(path)

    def writeSettingsForSingleSimulation(self, path):
        return self.write_settings_for_single_simulation(path)

    def initialize_custom_settings(self, filename):
        """
        Loads simulation-specific settings

        :param filename: {str} absolute path to the simulation-specific setting file
        :return: None
        """
        self.myCustomSettings, self.myCustomSettingsPath = load_settings(filename)

    def initializeCustomSettings(self, filename):
        return self.initialize_custom_settings(filename)

    def get_default_field_params(self):
        """
        Creates dictionary for FieldParams setting (specifying how to diplay given field).
        All field params have default values

        :return: {dict} Dictionary of field parameters (field visualzation parameters)
        """
        paramsDict = {}

        paramsDict["MinRange"] = self.get_setting("MinRange")
        paramsDict["MinRangeFixed"] = self.get_setting("MinRangeFixed")
        paramsDict["MaxRange"] = self.get_setting("MaxRange")
        paramsDict["MaxRangeFixed"] = self.get_setting("MaxRangeFixed")

        paramsDict["ShowPlotAxes"] = self.get_setting("ShowPlotAxes")

        paramsDict["NumberOfLegendBoxes"] = self.get_setting("NumberOfLegendBoxes")
        paramsDict["NumberAccuracy"] = self.get_setting("NumberAccuracy")
        paramsDict["LegendEnable"] = self.get_setting("LegendEnable")

        paramsDict["NumberOfContourLines"] = self.get_setting("NumberOfContourLines")

        paramsDict["ScaleArrowsOn"] = self.get_setting("ScaleArrowsOn")
        color = self.get_setting("ArrowColor")

        paramsDict["ArrowColor"] = color

        paramsDict["ArrowLength"] = self.get_setting("ArrowLength")
        paramsDict["FixedArrowColorOn"] = self.get_setting("FixedArrowColorOn")
        paramsDict["OverlayVectorsOn"] = self.get_setting("OverlayVectorsOn")
        paramsDict["ScalarIsoValues"] = self.get_setting("ScalarIsoValues")
        paramsDict["ContoursOn"] = self.get_setting("ContoursOn")

        paramsDict["AutomaticMovie"] = self.get_setting("AutomaticMovie")

        return paramsDict

    def getDefaultFieldParams(self):
        return self.get_default_field_params()

    def update_fields_params(self, field_name, field_dict):
        # todo - see if this function is needed
        """
        Called by ConfigurationDialog - stores field params dictionary (fieldDict) associated with field (fieldName)
        in the Configuration.simFieldsParams. Also stores this information in the settings file on the hard drive

        NOT SURE IF THIS FUNCTION IS ACTUALLY NEEDED

        :param field_name: {str} name of the field
        :param field_dict: {dict} dictionary of field-related visualization settings
        :return: None
        """
        fieldParamsDict = self.get_setting("FieldParams")
        self.simFieldsParams = fieldParamsDict

        fieldName = str(field_name)

        print('CONFIGURATION: fieldName =', field_name)
        print('CONFIGURATION: fieldDict =', field_dict)

        self.simFieldsParams[fieldName] = field_dict  # do regardless of in there or not

        self.set_setting('FieldParams', self.simFieldsParams)

        self.simFieldsParams[fieldName] = field_dict  # do regardless of in there or not

    def updateFieldsParams(self, fieldName, fieldDict):
        return self.update_fields_params(field_name=fieldName, field_dict=fieldDict)

    def get_setting(self, _key, field_name=None):
        """
        Retrieves setting name from the setting file(s)/database(s)

        :param _key: {str} name of the setting
        :param field_name:{str} name of the field - optional and used only in conjunction with dictionary of
        settings associated with field visualization parameters
        :return: {object} object corresponding to a given setting
        """
        if self.myCustomSettings:
            setting_storage = self.myCustomSettings

        else:
            setting_storage = self.myGlobalSettings

        # some settings are stored in the global settings e.g. number of recent simulations or recent simulations list
        if _key in self.globalOnlySettings:
            setting_storage = self.myGlobalSettings

        if field_name is not None:
            field_params = self.get_setting('FieldParams')
            try:
                single_field_params = field_params[field_name]

                return single_field_params[_key]
            except LookupError:
                pass  # returning global parameter for the field

        # val = settingStorage.getSetting(_key)
        # a way to fetch unknown setting from default setting and writing it back to the custom settins
        try:
            val = setting_storage.getSetting(_key)
        except KeyError:
            # attempt to fetch setting from global setting
            setting_storage = self.myGlobalSettings
            val = setting_storage.getSetting(_key)
            # and write it to custom setting
            if self.myCustomSettings:
                setting_storage = self.myCustomSettings
                setting_storage.setSetting(_key, val)

        return val

    # we append an optional fieldName now to allow for field-dependent parameters from Prefs
    def getSetting(self, _key, fieldName=None):
        return self.get_setting(_key, field_name=fieldName)

    def set_setting(self, _key, _value):
        """
        stores object (_value) under name (_key) in the relevant setting file(s)/databases(s)

        :param _key: {str} name of the setting
        :param _value: {object} object associated with the setting
        :return: None
        """
        val = _value

        # some settings are stored in the global settings e.g. number of recent simulations or recent simulations list
        if _key in self.globalOnlySettings:
            self.myGlobalSettings.setSetting(_key, val)

        elif _key in self.customOnlySettings:  # some settings are stored in the custom settings e.g. WindowsLayout

            if self.myCustomSettings:
                self.myCustomSettings.setSetting(_key, val)

        else:
            self.myGlobalSettings.setSetting(_key, val)

            if self.myCustomSettings:
                self.myCustomSettings.setSetting(_key, val)

    def setSetting(self, _key, _value):
        return self.set_setting(_key, _value)


def init_configuration() -> Configuration:
    """
    Function that "flushes" all configuration settings.

    :return: new configuration instance
    """
    return Configuration()


def initConfiguration():
    return init_configuration()
