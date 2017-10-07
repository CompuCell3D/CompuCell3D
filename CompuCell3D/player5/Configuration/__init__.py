# from PyQt4.QtGui import *
# from PyQt4.QtCore import *

# from Messaging import stdMsg, dbgMsg,pd, errMsg, setDebugging
# setDebugging(1)

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

from os import environ, path
import os

# determining icon's path

# _path = os.path.abspath(os.path.dirname(__file__))
#
# _path = os.path.abspath(os.path.join(_path+'../../../'))
#
# icons_dir = os.path.abspath(os.path.join(_path, 'player5/icons'))

# todo - handle bad file format for settings
# todo - at the beginning read all settings and manke sure there are not issues in the stored settings
# todo  - fix replaceCustomSettingsWithDefaults -  see how and where it is used

from SettingUtils import *
from SettingUtils import _global_setting_path

# the imports have to be fixed in the entire CC3D!!!!
try:
    from CompuCell3D.player5.Config.settingdict import SettingsSQL
except ImportError:
    from Config.settingdict import SettingsSQL



LATTICE_TYPES = {"Square": 1, "Hexagonal": 2}

maxNumberOfRecentFiles = 5

MODULENAME = '------- player5/Configuration/__init__.py: '


class Configuration():
    # # # defaultSettings = defaultSettings()
    defaultSettings, defaultSettingsPath = loadDefaultSettings()
    myGlobalSettings, myGlobalSettingsPath = loadGlobalSettings()

    # globalSettingsSQL = SettingsSQL('GlobalSetting.sqlite')
    # customSettingsSQL = SettingsSQL('CustomSetting.sqlite')

    synchronizeGlobalAndDefaultSettings(defaultSettings, myGlobalSettings, myGlobalSettingsPath)

    myCustomSettings = None  # this is an object that stores settings for custom settings i.e. ones which are associated with individual cc3d projects
    myCustomSettingsPath = ''

    globalOnlySettings = ['RecentSimulations', 'NumberOfRecentSimulations', 'OutputLocation', 'ProjectLocation']
    customOnlySettings = ['WindowsLayout']

    activeFieldNamesList = []


def initConfiguration():
    Configuration.defaultSettings, Configuration.defaultSettingsPath = loadDefaultSettings()
    Configuration.myGlobalSettings, Configuration.myGlobalSettingsPath = loadGlobalSettings()
    synchronizeGlobalAndDefaultSettings(Configuration.defaultSettings, Configuration.myGlobalSettings,
                                        Configuration.myGlobalSettingsPath)
    Configuration.myCustomSettings = None
    Configuration.myCustomSettingsPath = ''


def replaceCustomSettingsWithDefaults():
    defaultSettings, path = loadDefaultSettings()

    Configuration.myCustomSettings = defaultSettings
    writeSettings(Configuration.myCustomSettings, Configuration.myCustomSettingsPath)


# def getIconsDir():return icons_dir
#
# def getIconPath(icon_name):
#
#     return os.path.abspath(os.path.join(getIconsDir(),icon_name))

# TODO change
# def getSettingNameList():
#     return Configuration.myGlobalSettings.getSettingNameList()


def getSettingNameList():
    return Configuration.myGlobalSettings.names()


def setUsedFieldNames(fieldNamesList):
    Configuration.activeFieldNamesList = fieldNamesList
    fieldParams = getSetting('FieldParams')

    # purging unneded fields
    cleanedFieldParams = {}
    for fieldName in Configuration.activeFieldNamesList:
        try:

            cleanedFieldParams[fieldName] = fieldParams[fieldName]

        except KeyError:
            cleanedFieldParams[fieldName] = getDefaultFieldParams()
            # cleanedFieldParams[fieldName] = 

            pass

    # print 'cleanedFieldParams.keys() = ', cleanedFieldParams.keys()   
    # import time
    # time.sleep(2)

    # print 'cleanedFieldParams =', cleanedFieldParams

    setSetting('FieldParams', cleanedFieldParams)
    # sys.exit()

#TODO change
# def writeAllSettings():
#     # print 'Configuration.myGlobalSettings.typeNames = ', Configuration.myGlobalSettings.getTypeSettingDictDict().keys()
#     # print 'Configuration.myGlobalSettings. = ', Configuration.myGlobalSettings.getTypeSettingDictDict()
#
#     writeSettings(Configuration.myGlobalSettings, Configuration.myGlobalSettingsPath)
#     writeSettings(Configuration.myCustomSettings, Configuration.myCustomSettingsPath)

def writeAllSettings():
    pass

def writeSettingsForSingleSimulation(path):
    """
    Here we are creating settings for a single simulation or loading them if they already exist
    :param path: {src} abs path to local settings
    :return: None
    """
    if not os.path.isfile(path):
        copy_settings(src_setting_path=_global_setting_path(),dst_setting_dir=os.path.dirname(path))
        Configuration.myCustomSettings, Configuration.myCustomSettingsPath =  load_settings(path)

    # if not Configuration.myCustomSettings:
    #
    #     copy_settings(src_setting_path=_global_setting_path(),dst_setting_dir=os.path.dirname(path))
    else:
        Configuration.myCustomSettings, Configuration.myCustomSettingsPath =  load_settings(path)





#todo - original file
# def writeSettingsForSingleSimulation(path):
#     if Configuration.myCustomSettings:
#         writeSettings(Configuration.myCustomSettings, path)
#     else:
#         # in case there is no custom settings object we use global settings and write them as local ones
#         writeSettings(Configuration.myGlobalSettings, path)
#         # once we wrote them we have to read them in to initialize objects
#         initializeCustomSettings(path)
#         # readCustomFile(path)


def initializeCustomSettings(filename):
    Configuration.myCustomSettings,  Configuration.myCustomSettingsPath = loadSettings(filename)
    # Configuration.myCustomSettingsPath = os.path.abspath(filename)


# def setSimFieldsParams(fieldNames):
def getDefaultFieldParams():
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


def initFieldsParams(fieldNames):  # called from SimpleTabView once we know the fields
    # print 

    fieldParams = getSetting('FieldParams')
    # print 'fieldParams.keys()=',fieldParams.keys()
    # print '\n\n\nfieldParams=',fieldParams
    # sys.exit()
    for field in fieldNames:

        if field not in fieldParams.keys() and field != 'Cell_Field':
            fieldParams[field] = getDefaultFieldParams()

    Configuration.simFieldsParams = fieldParams
    # print 'initFieldsParams fieldParams = ',fieldParams
    setSetting('FieldParams', fieldParams)


def updateFieldsParams(fieldName, fieldDict):
    fieldParamsDict = getSetting("FieldParams")
    Configuration.simFieldsParams = fieldParamsDict

    fieldName = str(fieldName)

    print 'CONFIGURATION: fieldName =', fieldName
    print 'CONFIGURATION: fieldDict =', fieldDict

    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    setSetting('FieldParams', Configuration.simFieldsParams)

    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not


def getRecentSimulationsIncludingNewItem(simName):
    tmpList = getSetting('RecentSimulations')
    maxLength = getSetting('NumberOfRecentSimulations')

    currentStrlist = [x for x in tmpList]

    # inserting new element
    currentStrlist.insert(0, simName)

    # eliminating duplicates
    seen = set()
    seen_add = seen.add
    currentStrlist = [x for x in currentStrlist if not (x in seen or seen_add(x))]

    # print  'len(currentStrlist)=',len(currentStrlist),' maxLength=',maxLength   

    # ensuring that we keep only NumberOfRecentSimulations elements
    if len(currentStrlist) > maxLength:
        currentStrlist = currentStrlist[: - (len(currentStrlist) - maxLength)]
        # print 'maxLength=',maxLength
    # print 'currentStrlist=',currentStrlist


    # setSetting('RecentSimulations',currentStrlist)
    # val = currentStrlist
    # print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n currentStrlist=',currentStrlist       
    # import time
    # time.sleep(2)

    return currentStrlist


# we append an optional fieldName now to allow for field-dependent parameters from Prefs
def getSetting(_key, fieldName=None):
    # print '_key=',_key
    settingStorage = None

    if Configuration.myCustomSettings:
        settingStorage = Configuration.myCustomSettings

    else:
        settingStorage = Configuration.myGlobalSettings

    # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
    if _key in Configuration.globalOnlySettings:
        settingStorage = Configuration.myGlobalSettings

    if fieldName is not None:
        fieldParams = getSetting('FieldParams')
        try:
            singleFieldParams = fieldParams[fieldName]

            return singleFieldParams[_key]
        except LookupError:
            pass  # returning global parameter for the field

    val = settingStorage.getSetting(_key)
    return val

    # TODO changes
    if _key == 'WindowsLayout':

        # val = settingStorage.getSetting(_key)
        val = Configuration.customSettingsSQL.getSetting(_key)
        return val

    else:
        val = settingStorage.getSetting(_key)

    # handling field params request
    if fieldName is not None:
        fieldParams = getSetting('FieldParams')
        # print 'fieldParams=',fieldParams
        try:
            singleFieldParams = fieldParams[fieldName]

            return singleFieldParams[_key]
        except LookupError:
            pass  # returning global parameter for the field

    if val:
        return val.toObject()
    else:  # try getting this setting value from global settings

        val = Configuration.myGlobalSettings.getSetting(_key)
        if val:
            settingStorage.setSetting(val.name, val.value, val.type)  # set missing setting
            return val.toObject()
        else:  # finally try default settings

            val = Configuration.defaultSettings.getSetting(_key)
            #             settingStorage.setSetting(val.name , val.value , val.type) # set missing setting
            if val:
                settingStorage.setSetting(val.name, val.value, val.type)  # set missing setting
                return val.toObject()

    return None  # if no setting is found return None


def setSetting(_key, _value):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs

    # print 'SETTING _key=',_key

    val = _value

    if _key == 'RecentSimulations':  # need to find better solution for that...
        simName = _value
        val = getRecentSimulationsIncludingNewItem(simName)  # val is the value that is stored int hte settings

    if _key in Configuration.globalOnlySettings:  # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
        # Configuration.myGlobalSettings.setSetting(_key, val) # TODO changes

        # Configuration.globalSettingsSQL.setSetting(_key, val)
        Configuration.myGlobalSettings.setSetting(_key, val)

    elif _key in Configuration.customOnlySettings:  # some settings are stored in the custom settings e.g. WindowsLayout

        if Configuration.myCustomSettings:
            # Configuration.myCustomSettings.setSetting(_key, val) # TODO changes

            # Configuration.customSettingsSQL.setSetting(_key, val)
            Configuration.myCustomSettings.setSetting(_key, val)

    else:
        #TODO change
        # Configuration.myGlobalSettings.setSetting(_key, val)  # we always save everythign to the global settings

        # Configuration.globalSettingsSQL.setSetting(_key, val)

        Configuration.myGlobalSettings.setSetting(_key, val)

        if Configuration.myCustomSettings:
            # TODO change
            # Configuration.myCustomSettings.setSetting(_key, val)
            # Configuration.customSettingsSQL.setSetting(_key, val)
            Configuration.myCustomSettings.setSetting(_key, val)

def syncPreferences():  # this function invoked when we close the Prefs dialog with the "OK" button
    pass
