from utils.global_imports import *


class Configuration:
    def __init__(self, _settings):

        self.settings = _settings

        # default settings
        self.defaultConfigs = {}
        self.defaultConfigs["SkipCommentsInPythonSnippets"] = False

        self.modifiedKeyboardShortcuts = {}  # dictionary actionName->shortcut for modified keyboard shortcuts - only reassinged shortcuts are stored

        self.initSyncSettings()

    def setting(self, _key):
        if _key in ["SkipCommentsInPythonSnippets"]:  # Boolean values
            val = self.settings.value(_key)
            val = self.settings.value(_key)
            if val:
                return val
            else:
                return self.defaultConfigs[_key]

    def setSetting(self, _key, _value):
        if _key in ["SkipCommentsInPythonSnippets"]:  # Boolean values
            self.settings.setValue(_key, QVariant(_value))

    def initSyncSettings(self):
        for key in self.defaultConfigs.keys():

            val = self.settings.value(key)
            if not val:
                self.setSetting(key, self.defaultConfigs[key])
