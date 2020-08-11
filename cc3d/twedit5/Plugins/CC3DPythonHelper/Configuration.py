from twedit.utils.global_imports import *


class Configuration:

    def __init__(self, _settings):

        self.settings = _settings

        # default settings

        self.defaultConfigs = {}

        self.defaultConfigs["SkipCommentsInPythonSnippets"] = False

        # dictionary actionName->shortcut for modified keyboard shortcuts - only reassinged shortcuts are stored
        self.modifiedKeyboardShortcuts = {}

        self.initSyncSettings()

    def setting(self, _key):

        if _key in ["SkipCommentsInPythonSnippets"]:  # Boolean values

            val = self.check_bool(self.settings.value(_key))

            if val:

                return val

            else:

                return self.defaultConfigs[_key]

    def setSetting(self, _key, _value):

        if _key in ["SkipCommentsInPythonSnippets"]:  # Boolean values

            self.settings.setValue(_key, QVariant(_value))

    def initSyncSettings(self):

        for key in list(self.defaultConfigs.keys()):

            val = self.check_bool(self.settings.value(key))

            if val is None:
                self.setSetting(key, self.defaultConfigs[key])

    def check_bool(self, _val):
        """
        Deals with a possible bug in Qt (occurs surprisingly often) and converts value read by QSettings
        object to boolean. The value read could be unicode or boolean
        :param _val: value returned by QSettings.value() fcn
        :return: {bool}
        """
        val_dict = {'true': True, 'false': False}
        if isinstance(_val, str) and _val in val_dict.keys():
            return val_dict[_val]
        else:
            return _val
