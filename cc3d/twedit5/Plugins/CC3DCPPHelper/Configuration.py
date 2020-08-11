from cc3d.twedit5.twedit.utils.global_imports import *


class Configuration:

    def __init__(self, _settings):
        # reference to settings object from Twedit main app
        self.settings = _settings

        #
        # default settings

        self.defaultConfigs = {}

        self.defaultConfigs["RecentModuleDirectory"] = ''

        #
        # dictionary actionName->shortcut for modified keyboard shortcuts.
        # only reassigned shortcuts are stored
        self.modifiedKeyboardShortcuts = {}
        self.initSyncSettings()

    def setting(self, _key):

        if _key in ["RecentModuleDirectory"]:
            val = self.settings.value(_key)

            if val:
                return val
            else:
                return self.defaultConfigs[_key]


    def setSetting(self, _key, _value):

        if _key in ["RecentModuleDirectory"]:  # string values
            self.settings.setValue(_key, QVariant(_value))

    def initSyncSettings(self):

        for key in list(self.defaultConfigs.keys()):
            val = self.settings.value(key)

            if not val:
                self.setSetting(key, self.defaultConfigs[key])
