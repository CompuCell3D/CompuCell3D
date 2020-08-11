from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, errMsg, setDebugging
import sys

(ORGANIZATION, APPLICATION) = ("SwatInternationalProductions", "Twedit5++")


class Configuration:

    def __init__(self):

        self.defaultConfigs = {}
        self.initialize_default_settings()

        # dictionary actionName->shortcut for modified keyboard shortcuts - only reassinged shortcuts are stored
        self.modifiedKeyboardShortcuts = {}

        # dictionary pluginName->autoLoad for modified plugin autoload data - only reassinged data are stored
        self.modifiedPluginsAutoloadData = {}

        if sys.platform.startswith('win'):

            self.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, APPLICATION)

        else:
            # note: on OSX settings are being cached so even if we delete settings file
            # in order for settings to be reset we need to logout from the account
            # https://stackoverflow.com/questions/19742452/qsettings-on-os-x-10-9-unable-to-locate-clear-it
            self.settings = QSettings(QSettings.NativeFormat, QSettings.UserScope, ORGANIZATION, APPLICATION)

        self.initSyncSettings()

        self.updatedConfigs = {}

    def reset_settings(self):
        """
        Resets settings to their default values
        :return:
        """

        for key, value in self.defaultConfigs.items():
            self.setSetting(_key=key, _value=value)

    def initialize_default_settings(self):
        """
        Initializes settings to their default values
        :return:
        """
        self.defaultConfigs["TabSpaces"] = 4

        self.defaultConfigs["UseTabSpaces"] = True

        self.defaultConfigs["DisplayLineNumbers"] = True

        self.defaultConfigs["FoldText"] = True

        self.defaultConfigs["TabGuidelines"] = True

        self.defaultConfigs["DisplayWhitespace"] = False

        self.defaultConfigs["DisplayEOL"] = False

        self.defaultConfigs["WrapLines"] = False

        self.defaultConfigs["ShowWrapSymbol"] = False

        self.defaultConfigs["DontShowWrapLinesWarning"] = False

        self.defaultConfigs["RestoreTabsOnStartup"] = False

        self.defaultConfigs["EnableAutocompletion"] = False

        self.defaultConfigs["EnableQuickTextDecoding"] = True

        self.defaultConfigs["AutocompletionThreshold"] = 2

        self.defaultConfigs["PanelSplitterState"] = QByteArray()

        self.defaultConfigs["InitialSize"] = QSize(600, 600)

        self.defaultConfigs["InitialPosition"] = QPoint(0, 0)

        self.defaultConfigs["RecentDocuments"] = []

        self.defaultConfigs["RecentDirectories"] = []

        self.defaultConfigs["ListOfOpenFiles"] = []

        self.defaultConfigs["ListOfOpenFilesAndPanels"] = []

        # FR stands for Find & Replace
        self.defaultConfigs["FRFindHistory"] = []

        self.defaultConfigs["FRReplaceHistory"] = []

        self.defaultConfigs["FRFiltersHistory"] = []

        self.defaultConfigs["FRDirectoryHistory"] = []

        self.defaultConfigs["FRInSelection"] = False

        self.defaultConfigs["FRInAllSubfolders"] = False

        self.defaultConfigs["FRSyntaxIndex"] = 0

        self.defaultConfigs["FRTransparencyEnable"] = True

        self.defaultConfigs["FROnLosingFocus"] = True

        self.defaultConfigs["FRAlways"] = False

        self.defaultConfigs["FROpacity"] = 75

        self.defaultConfigs["ZoomRange"] = 4
        self.defaultConfigs["ZoomRangeFindDisplayWidget"] = 0

        # if sys.platform.startswith('darwin'):
        #     self.defaultConfigs["ZoomRange"] = 0

        # index of the current tab  - 1 means we should make last open tab current
        self.defaultConfigs["CurrentTabIndex"] = -1

        # index of the current panel  - 1 means we should make last open tab current
        self.defaultConfigs["CurrentPanelIndex"] = 0

        self.defaultConfigs["KeyboardShortcuts"] = []

        self.defaultConfigs["PluginAutoloadData"] = []

        self.defaultConfigs["BaseFontName"] = "Courier New"

        self.defaultConfigs["BaseFontSize"] = "10"

        self.defaultConfigs["Theme"] = "Default"

    @staticmethod
    def to_bool(val):
        """
        Deals with a possible bug in Qt (occurs surprisingly often) and converts value read by QSettings

        object to boolean. The value read could be unicode or boolean


        :param val: value returned by QSettings.value() fcn

        :return: {Boolean}

        """

        val_dict = {'true': True, 'false': False}

        if isinstance(val, str):

            return val_dict[val.strip().lower()]

        else:

            return val

    def setting(self, _key):

        # Boolean values
        if _key in ["UseTabSpaces", "DisplayLineNumbers", "FoldText",  "TabGuidelines", "DisplayWhitespace",

                    "DisplayEOL", "WrapLines", "ShowWrapSymbol", "DontShowWrapLinesWarning",

                    "RestoreTabsOnStartup", "EnableAutocompletion", "EnableQuickTextDecoding", "FRInSelection",

                    "FRInAllSubfolders", "FRTransparencyEnable", "FROnLosingFocus", "FRAlways"]:

            variant = self.settings.value(_key)
            if variant is not None:

                return self.to_bool(variant)
            else:
                return self.defaultConfigs[_key]

        elif _key in ["BaseFontSize", "BaseFontName", "Theme"]:

            val = self.settings.value(_key)

            if val is not None:

                return val

            else:

                return self.defaultConfigs[_key]

        elif _key in ["TabSpaces", "ZoomRange", "ZoomRangeFindDisplayWidget", "AutocompletionThreshold",

                      "FRSyntaxIndex", "FROpacity", "CurrentTabIndex", "CurrentPanelIndex"]:  # integer values

            variant = self.settings.value(_key)
            if variant is not None:

                # toInt returns tuple and first element of if is the integer the second one is flag
                return int(variant)

            else:

                return self.defaultConfigs[_key]

        # QSize values
        elif _key in ["InitialSize"]:

            val = self.settings.value(_key)

            if val.isValid():

                return val

            else:

                return self.defaultConfigs[_key]

        # QPoint values
        elif _key in ["InitialPosition"]:

            val = self.settings.value(_key)

            if val:

                return val

            else:

                return self.defaultConfigs[_key]

        # QStringList values
        elif _key in ["RecentDocuments", "RecentDirectories", "ListOfOpenFiles", "ListOfOpenFilesAndPanels",

                      "FRFindHistory", "FRReplaceHistory", "FRFiltersHistory", "FRDirectoryHistory",

                      "KeyboardShortcuts", "PluginAutoloadData"]:

            val = self.settings.value(_key)

            if val:

                return val

            else:

                return self.defaultConfigs[_key]

        # QByteArray values
        elif _key in ["PanelSplitterState"]:

            val = self.settings.value(_key)

            if val:

                return val

            else:

                return self.defaultConfigs[_key]

    def setSetting(self, _key, _value):

        # Boolean values
        if _key in ["UseTabSpaces", "DisplayLineNumbers", "FoldText", "TabGuidelines", "DisplayWhitespace",
                    "DisplayEOL", "WrapLines", "ShowWrapSymbol", "DontShowWrapLinesWarning",
                    "RestoreTabsOnStartup", "EnableAutocompletion", "EnableQuickTextDecoding", "FRInSelection",
                    "FRInAllSubfolders", "FRTransparencyEnable", "FROnLosingFocus", "FRAlways"]:

            self.settings.setValue(_key, QVariant(_value))

        # integer values
        elif _key in ["TabSpaces", "ZoomRange", "ZoomRangeFindDisplayWidget", "AutocompletionThreshold",
                      "FRSyntaxIndex", "FROpacity", "CurrentTabIndex", "CurrentPanelIndex"]:

            self.settings.setValue(_key, _value)

        # string values
        elif _key in ["BaseFontName", "BaseFontSize", "Theme"]:

            self.settings.setValue(_key, QVariant(_value))

        # QSize, QPoint,QStringList , QString values
        elif _key in ["RecentDocuments", "RecentDirectories", "InitialSize", "InitialPosition", "ListOfOpenFiles",

                      "ListOfOpenFilesAndPanels", "FRSyntax", "FRFindHistory", "FRReplaceHistory", "FRFiltersHistory",

                      "FRDirectoryHistory", "KeyboardShortcuts",

                      "PluginAutoloadData"]:

            self.settings.setValue(_key, QVariant(_value))

        # QByteArray
        elif _key in ["PanelSplitterState"]:

            self.settings.setValue(_key, QVariant(_value))

        else:

            dbgMsg("Wrong format of configuration option:", _key, ":", _value)

    def setPluginAutoloadData(self, _pluginName, _autoloadFlag):

        self.modifiedPluginsAutoloadData[_pluginName] = _autoloadFlag

    def pluginAutoloadData(self):

        return self.modifiedPluginsAutoloadData

    def preparePluginAutoloadDataForStorage(self):

        self.modifiedPluginsDataStringList = []

        for pluginName in list(self.modifiedPluginsAutoloadData.keys()):
            self.modifiedPluginsDataStringList.append(pluginName)

            self.modifiedPluginsDataStringList.append(

                str(self.modifiedPluginsAutoloadData[pluginName]))  # converting bool to string

        print('STORING ', self.modifiedPluginsAutoloadData)

        for i in range(0, len(self.modifiedPluginsDataStringList), 2):
            print((str(self.modifiedPluginsDataStringList[i]), str(self.modifiedPluginsDataStringList[i + 1])))

        self.setSetting("PluginAutoloadData", self.modifiedPluginsDataStringList)

    def setKeyboardShortcut(self, _actionName, _keyboardshortcut):

        self.modifiedKeyboardShortcuts[_actionName] = _keyboardshortcut

    def get_keyboard_shortcut(self, action_name):
        try:
            return self.modifiedKeyboardShortcuts[action_name]
        except KeyError:
            return ''

    def keyboardShortcuts(self):

        return self.modifiedKeyboardShortcuts

    def prepareKeyboardShortcutsForStorage(self):

        self.modifiedKeyboardShortcutsStringList = []

        for actionName in list(self.modifiedKeyboardShortcuts.keys()):
            self.modifiedKeyboardShortcutsStringList.append(actionName)

            self.modifiedKeyboardShortcutsStringList.append(self.modifiedKeyboardShortcuts[actionName])

        self.setSetting("KeyboardShortcuts", self.modifiedKeyboardShortcutsStringList)

    def initSyncSettings(self):

        for key in list(self.defaultConfigs.keys()):

            val = self.settings.value(key)

            # if not val.isValid():

            if not val:
                self.setSetting(key, self.defaultConfigs[key])

        # initialize self.modifiedKeyboardShortcuts

        self.modifiedKeyboardShortcutsStringList = self.setting("KeyboardShortcuts")

        for i in range(0, len(self.modifiedKeyboardShortcutsStringList), 2):
            self.modifiedKeyboardShortcuts[str(self.modifiedKeyboardShortcutsStringList[i])] = str(

                self.modifiedKeyboardShortcutsStringList[i + 1])

        self.modifiedPluginsDataStringList = self.setting("PluginAutoloadData")

        print('INIT SYNC self.modifiedPluginsDataStringList=', self.modifiedPluginsDataStringList)

        for i in range(0, len(self.modifiedPluginsDataStringList), 2):
            print((str(self.modifiedPluginsDataStringList[i]), str(self.modifiedPluginsDataStringList[i + 1])))

            self.modifiedPluginsAutoloadData[str(self.modifiedPluginsDataStringList[i])] = (

                    str(self.modifiedPluginsDataStringList[i + 1]).rstrip() == 'True')  # converting string to bool
