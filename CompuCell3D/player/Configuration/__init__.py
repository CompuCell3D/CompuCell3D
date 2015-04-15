from PyQt4.QtGui import *
from PyQt4.QtCore import *

#from Messaging import stdMsg, dbgMsg,pd, errMsg, setDebugging
#setDebugging(1)

from os import environ,path
import os

from SettingUtils import Setting, CustomSettings, writeSettings, loadSettings, defaultSettings

#(ORGANIZATION, APPLICATION) = ("Biocomplexity", "PyQtPlayerNew")

(SETTINGS_FOLDER, SETTINGS_FILE_NAME) = (".compucell3d", "_settings.xml")
LATTICE_TYPES = {"Square":1,"Hexagonal":2}

maxNumberOfRecentFiles=5

MODULENAME = '------- player/Configuration/__init__.py: '


def loadGlobalSettings():
    
    global_setting_dir = os.path.abspath(os.path.join(os.path.expanduser('~') , SETTINGS_FOLDER))
    global_setting_path = os.path.abspath(os.path.join(global_setting_dir , SETTINGS_FILE_NAME)) # abspath normalizes path
    # print 'LOOKING FOR global_setting_path=',global_setting_path
    
    #create global settings  directory inside user home directory
    if not os.path.isdir(global_setting_dir):
        try:
            os.makedirs(global_setting_dir)
    
        except:
            print 'Cenfiguration: COuld not make directory: ',global_setting_dir, ' to store global settings. Please make sure that you have appropriate write permissions'
            import sys
            sys.exit()
    
    globalSettings = loadSettings (global_setting_path)    
    if not globalSettings:
        globalSettings, default_setting_path = loadDefaultSettings()
        globalSettings.saveAsXML(global_setting_path)        
        
        return globalSettings , global_setting_path     
    return  globalSettings , global_setting_path            

def loadDefaultSettings():
       
    default_setting_path = os.path.abspath(os.path.join(os.path.dirname(__file__) , SETTINGS_FILE_NAME)) # abspath normalizes path

    defaultSettings = loadSettings (default_setting_path)    
    if not defaultSettings:
        return None, None
        
    return defaultSettings , default_setting_path
    
    
#this function checks for new settings in the default settings file    
def synchronizeGlobalAndDefaultSettings(defaultSettings,globalSettings,globalSettingsPath):
    defaultSettingsNameList = defaultSettings.getSettingNameList()
    globalSettingsNameList = globalSettings.getSettingNameList()
    
    newSettingNames = set(defaultSettingsNameList) - set(globalSettingsNameList)
    
    for newSettingName in newSettingNames:
        print 'newSettingName = ', newSettingName
        newDefaultSetting = defaultSettings.getSetting(newSettingName)
        
        globalSettings.setSetting(newDefaultSetting.name , newDefaultSetting.value , newDefaultSetting.type)
        writeSettings (globalSettings,globalSettingsPath) 
    # sys.exit()
    
        
class Configuration():
    
    defaultSettings , defaultSettingsPath = loadDefaultSettings()    
    myGlobalSettings , myGlobalSettingsPath = loadGlobalSettings()       
    
    synchronizeGlobalAndDefaultSettings( defaultSettings , myGlobalSettings , myGlobalSettingsPath )
    
    myCustomSettings = None # this is an object that stores settings for custom settings i.e. ones which are associated with individual cc3d projects
    myCustomSettingsPath = ''
    
    globalOnlySettings = ['RecentSimulations','NumberOfRecentSimulations']
    
    activeFieldNamesList = []
    
    # # # defaultSettings = defaultSettings()
    
    
def getSettingNameList(): return Configuration.myGlobalSettings.getSettingNameList()
        
        
def setUsedFieldNames(fieldNamesList):
    Configuration.activeFieldNamesList = fieldNamesList
    fieldParams = getSetting('FieldParams')
    
    
    
    #purging unneded fields    
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
    
    setSetting('FieldParams',cleanedFieldParams)
    # sys.exit()
    


def writeAllSettings():

    # print 'Configuration.myGlobalSettings.typeNames = ', Configuration.myGlobalSettings.getTypeSettingDictDict().keys()
    # print 'Configuration.myGlobalSettings. = ', Configuration.myGlobalSettings.getTypeSettingDictDict()
    
    writeSettings(Configuration.myGlobalSettings , Configuration.myGlobalSettingsPath)
    writeSettings(Configuration.myCustomSettings , Configuration.myCustomSettingsPath)
    
def writeSettingsForSingleSimulation(path):
    if Configuration.myCustomSettings:        
        writeSettings(Configuration.myCustomSettings,path)
    else:
        #in case there is no custom settings object we use global settings and write them as local ones 
        writeSettings(Configuration.myGlobalSettings,path)
        # once we wrote them we have to read them in to initialize objects
        readCustomFile(path)


def readCustomFile(fileName):
        
    import XMLUtils
    xml2ObjConverter = XMLUtils.Xml2Obj()

    fileFullPath = os.path.abspath(fileName)
    cs = CustomSettings()        
    cs.readFromXML(fileFullPath)
    # # # FIX HERE
    Configuration.myCustomSettings = cs
    Configuration.myCustomSettingsPath = fileName

    
#def setSimFieldsParams(fieldNames):
def getDefaultFieldParams():
    paramsDict = {}

    paramsDict["MinRange"] = getSetting("MinRange")
    paramsDict["MinRangeFixed"] = getSetting("MinRangeFixed")
    paramsDict["MaxRange"] = getSetting("MaxRange")
    paramsDict["MaxRangeFixed"] = getSetting("MaxRangeFixed")
    
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
    
    return paramsDict

def initFieldsParams(fieldNames):   # called from SimpleTabView once we know the fields
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
    setSetting('FieldParams',fieldParams )
    
def updateFieldsParams(fieldName,fieldDict):
    
    fieldParamsDict = getSetting("FieldParams")
    Configuration.simFieldsParams = fieldParamsDict

    fieldName = str(fieldName)
    
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    
    
    setSetting('FieldParams',Configuration.simFieldsParams)
    
    
        
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    
    
def getRecentSimulationsIncludingNewItem(simName):

    tmpList = getSetting('RecentSimulations')
    maxLength = getSetting('NumberOfRecentSimulations')
    
    currentStrlist = [x for x in tmpList]
    
    #inserting new element   
    currentStrlist.insert(0,simName)
    
    #eliminating duplicates        
    seen = set()
    seen_add = seen.add
    currentStrlist = [ x for x in currentStrlist if not (x in seen or seen_add(x))]        
    
    # print  'len(currentStrlist)=',len(currentStrlist),' maxLength=',maxLength   
    
    # ensuring that we keep only NumberOfRecentSimulations elements
    if len(currentStrlist) > maxLength:
        currentStrlist = currentStrlist[: - ( len(currentStrlist)-maxLength ) ] 
    # print 'maxLength=',maxLength    
    # print 'currentStrlist=',currentStrlist
        
    
    # setSetting('RecentSimulations',currentStrlist)
    # val = currentStrlist
    # print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n currentStrlist=',currentStrlist       
    # import time
    # time.sleep(2)
    
    return currentStrlist
    
    
def getSetting(_key, fieldName=None):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        
    # print '_key=',_key
    settingStorage = None    
    
    if Configuration.myCustomSettings:
        settingStorage = Configuration.myCustomSettings
    else:
        settingStorage = Configuration.myGlobalSettings
        
    if _key in Configuration.globalOnlySettings: # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
        settingStorage = Configuration.myGlobalSettings
    
    val = settingStorage.getSetting(_key)  
    
    #handling field params request
    if fieldName is not None:
        fieldParams = getSetting('FieldParams')
        try:
            singleFieldParams = fieldParams[fieldName]
            return singleFieldParams[_key] 
        except LookupError:
            pass # returning global parameter for the field
    
    if val:
        return val.toObject()
    else: # try getting this setting value from global settings
        
        val = Configuration.myGlobalSettings.getSetting(_key)  
        if val:  
            settingStorage.setSetting(val.name , val.value , val.type) # set missing setting
            return val.toObject()
        else:#finally try default settings
            
            val = Configuration.defaultSettings.getSetting(_key)  
            settingStorage.setSetting(val.name , val.value , val.type) # set missing setting
            if val:  
                return val.toObject()
                
        
    return None # if no setting is found return None
        
def setSetting(_key,_value):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        
    # print 'SETTING _key=',_key
        
    val = _value
    
    if _key == 'RecentSimulations' : # need to find better solution for that... 
        simName = _value
        val  = getRecentSimulationsIncludingNewItem(simName) # val is the value that is stored int hte settings
    
    
    if _key in Configuration.globalOnlySettings: # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
        Configuration.myGlobalSettings.setSetting(_key,val)  
    else:    
        Configuration.myGlobalSettings.setSetting(_key,val)  # we always save everythign to the global settings
        if Configuration.myCustomSettings:
            Configuration.myCustomSettings.setSetting(_key,val)
    

def syncPreferences():   # this function invoked when we close the Prefs dialog with the "OK" button
    pass

    
