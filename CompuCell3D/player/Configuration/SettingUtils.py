from PyQt4.QtGui import *
from PyQt4.QtCore import *
from os import environ,path
import os

from DefaultSettingsData import *

def writeSettings (settingsObj,path):
    if settingsObj:
        settingsObj.saveAsXML(path)   

def loadSettings(filename):

    if os.path.isfile(filename):
        import XMLUtils
        xml2ObjConverter = XMLUtils.Xml2Obj()

        fileFullPath = os.path.abspath(filename)
        settings = CustomSettings()        
        settings.readFromXML(filename)
        
        return settings
        
    return None 

def get_global_setting_path():
    """
    returns global settings path
    :return: None
    """
    global_setting_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), SETTINGS_FOLDER))
    global_setting_path = os.path.abspath(os.path.join(global_setting_dir, SETTINGS_FILE_NAME))
    return global_setting_path

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
        
class Setting(object):
    storedType2XML = {
    'int':'self.int2XML',
    'float':'self.float2XML',
    'bool':'self.bool2XML',
    'str':'self.str2XML',
    'strlist':'self.strlist2XML',
    'color':'self.color2XML',
    'dict':'self.dict2XML',
    # 'typecolormap':'self.typecolormap2XML',
    'size':'self.size2XML',
    'point':'self.point2XML',
    'floatlist':'self.floatlist2XML',
    'intlist':'self.intlist2XML',
    'bytearray':'self.bytearray2XML'}


    XML2StoredType = {
    'int':'self.XML2Int',
    'float':'self.XML2Float',
    'bool':'self.XML2Bool',
    'str':'self.XML2Str',
    'strlist':'self.XML2Strlist',
    'color':'self.XML2Color',
    'dict':'self.XML2Dict',
    # 'typecolormap':'self.typecolormap2XML',
    'size':'self.XML2Size',
    'point':'self.XML2Point',
    'floatlist':'self.XML2Floatlist',
    'intlist':'self.XML2Intlist',    
    'bytearray':'self.XML2ByteArray'}

    
    # ,'bool':'bool'
    def __init__(self, _name, _value, _type):
        self.__name = _name
        self.__value = _value        
        self.__type = _type
        
    @property    
    def name(self):
        return self.__name
        
    @name.setter    
    def name(self,_name):
        self.__name = _name

    @property    
    def type(self):
        return self.__type
        
    @type.setter    
    def type(self,_type):
        self.__type = _type

    @property    
    def value(self):
        return self.__value
        
    @value.setter    
    def value(self,_value):
        self.__value = _value
        
    def __str__(self):
        return str(self.__name)+':['+str(self.__value)+','+str(self.__type)+']'
        
    def __repr__(self):
        return self.__str__()
        
    def initNameType(self,element):
        self.name = element.getAttribute('Name')
        self.type = element.getAttribute('Type')
        
    def int2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},str(self.value))
        
    def XML2Int(self,element):
        self.initNameType(element)
        self.value = int(element.cdata)        
        
        
    def float2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},str(self.value))
        
    def XML2Float(self,element):
        self.initNameType(element)
        self.value = float(element.cdata)        
        
    def bool2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},str(self.value))
        
    def XML2Bool(self,element):
        self.initNameType(element)
        
        self.value = True if str (element.cdata) == 'True' else False       

    def str2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},str(self.value))

    def XML2Str(self,element):
        self.initNameType(element)
        
        self.value = str (element.cdata) 
                
    def color2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},self.QColorToString(self.value))

    def XML2Color(self,element):
        self.initNameType(element)        
        self.value = QColor(str(element.cdata))

        
    def size2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},self.QSizeToString(self.value))
        
    def XML2Size(self,element):
        self.initNameType(element)              
        self.value = self.StringToQSize(element.cdata)
        
    def point2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},self.QPointToString(self.value))
        
    def XML2Point(self,element):
        self.initNameType(element)              
        self.value = self.StringToQPoint(element.cdata)

    def floatlist2XML(self,parentElement):        
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type}, ','.join(map(str,self.value)))
        
    def XML2Floatlist(self,element):
        self.initNameType(element)        
        strlist = element.cdata.split(',')
        # print 'strlist=',strlist
        if not len (strlist) or  not strlist[0]:        
            self.value = []            
        else:
            self.value = map(float , strlist)
        
        
    def intlist2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type}, map(str,self.value).join(','))
        
    def XML2Intlist(self,element):
        self.initNameType(element)        
        strlist = element.cdata.split(',')
        if not len (strlist) or  not strlist[0]:        
            self.value = []            
        else:
            self.value = map(int , strlist)
        
    def bytearray2XML(self,parentElement):
        parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type},self.QByteArrayToString(self.value))

    def XML2ByteArray(self,element):
        self.initNameType(element)              
        self.value = self.StringToQByteArray(element.cdata)
        
    #strlist is considered primitive data type for settings  - just like e.g. QColor
    def strlist2XML(self,parentElement):
        
        strlistContainerElem = parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type})
        # print 'strlist2XML self.value = ', self.value
        # print 'trlist2XML type self.value  = ', type(self.value)
        for string in self.value:
            # print 'string = ', string
            strlistContainerElem.ElementCC3D('e',{'Name':'e','Type':'str'},string)
        
        
    def XML2Strlist(self,element):
        self.initNameType(element)

        import XMLUtils
        elementList = XMLUtils.CC3DXMLListPy(element.getElements(""))
        self.value = []        
        for el in elementList:                        
            s = Setting('e',None,'str')        
            s.fromXML(el)
            self.value.append(s.value)



            
    def dict2XML(self,parentElement):
        
        dictContainerElem = parentElement.ElementCC3D('e',{'Name':self.name,'Type':self.type})
        for key , setting in sorted(self.value.iteritems()): # sorting dict by keys
            # if key=='FGF':
            #     try:
            #         print 'dict elem=', key, ' setting =',setting
            #     except:
            #         pass

            setting.toXML(dictContainerElem)
            
            # strlistContainerElem.ElementCC3D('e',{'Type':'str'},string)
        # sys.exit()
        
    def XML2Dict(self,element):
        # # # # Dictionaries are stored Settings (name, {...}, type = 'pythondict') in self.value
        # # # self.initNameType(element)
        # # # self.value = Setting('pythondict',{},'pythondict')
        
        self.initNameType(element)              
        self.value = {}
        dictionary = self.value
        
        import XMLUtils
        elementList = XMLUtils.CC3DXMLListPy(element.getElements(""))
        
        for el in elementList:            
            s = Setting(el.getAttribute('Name'),None,el.getAttribute('Type'))
            s.fromXML(el)
            dictionary [el.getAttribute('Name')] = s
        
    # def typecolormap2XML(self,parentElement):
        
        # strlistContainerElem = parentElement.ElementCC3D(self.name,{'Type':self.type},self.TypeColorMapToString(self.value))
        
        
        
    def toXML(self,parentElement):
        # print 'self.type = ',self.type, 'self.name = ',self.name
        # print 'Setting.storedType2XML[self.type]=',Setting.storedType2XML[self.type]
        try:
            eval(Setting.storedType2XML[self.type]+'(parentElement)')
        except KeyError:
            print 'COULD NOT PROCESS seting name = ',self.name, ' type=', self.type 
        
    def fromXML(self,parentElement):
        # print 'Setting.storedType2XML[self.type]=',Setting.storedType2XML[self.type]
        eval(Setting.XML2StoredType[self.type]+'(parentElement)')
        
        
    def StringToQSize(self,_sizeStr):
        sizeList = _sizeStr.split(',')
        sizeListInt =  map (int, sizeList)
        
        return QSize(sizeListInt[0],sizeListInt[1])

    def StringToQPoint(self,_pointStr):
        pointList = _pointStr.split(',')
        pointListInt =  map (int, pointList)
        
        return QPoint(pointListInt[0],pointListInt[1])

        
    def StringToQByteArray(self,_elemsStr):
        try:
            elemsList = map( chr, map (int, _elemsStr.split(',') ) )
        except:
            print 'CONFIGURATIN: COULD NOT CONVERT STEING TO BYTEARRAY'
            elemsList = []

        ba = QByteArray()
        for i in xrange(len(elemsList)):
            ba.append(elemsList[i])
        
        return ba

    
    def QColorToString(self, _val):
    
        return str(_val.name())
        
    def QPointToString(self, _val):
    
        return str(_val.x())+','+str(_val.y())

    def QSizeToString(self, _val):
    
        return str(_val.width())+','+str(_val.height())
        
    def QByteArrayToString(self, _val):
    
        out_str = ''
        for i in range(_val.count()):
            out_str += str(ord(_val[i]))
            if i<_val.count()-1:
                out_str += ','    
        return out_str

    def TypeColorMapToString(self, _val):
        out_str = ''
        for key, setting in _val.iteritems():
            out_str += str(key)+','+str(setting.value.name())+','
            
        return out_str[:-1]            

    def typeColorMap2Setting(self, typeColorMap):        
        
        for key,val in typeColorMap.iteritems():
            if type(val) == type (Setting(None,None,None)):
                #typeColorMap is in the proper format
                return typeColorMap
            else:
                #typeColorMap needs to be converted to a proper format
                break
                    
        typeColorMapSetting = {}
        
        for key,val in typeColorMap.iteritems():
            typeColorMapSetting [str(key)] = Setting(str(key),val,'color')    
        
        return typeColorMapSetting
        
    def toTypeColorMap(self):
        typeColorMap = {}
        # print 'self.value=',self.value
        # sys.exit()
        for key,val in self.value.iteritems():
            
            typeColorMap [int(key)] =  val.value
            
        # print 'typeColorMap=',typeColorMap    
        
        return typeColorMap
        
    def windowsLayout2Setting(self, windowsLayout):
        try:
            # print 'windowsLayout=',windowsLayout
            # print 'type(windowsLayout.items()[0][0]) = ',type(windowsLayout.items()[0][1])
            # print ' type(Setting) = ',type(Setting)
            if type(windowsLayout.items()[0][1]) == type(Setting(None,None,None)):
                return windowsLayout
        except IndexError:        
            pass
    
        # for key,val in windowsLayout.iteritems():
            # if type(val) == type (Setting(None,None,None)):
                # #fieldParams is in the proper format
                # return windowsLayout
            # else:
                # #fieldParams needs to be converted to a proper format
                # break   
    
        windowsLayoutSetting = {}
        
        for windowName, singleWindowDict in windowsLayout.iteritems():
        
            windowsLayoutSetting [windowName] = Setting(windowName,{},'dict')
                        
            
            singleWindowsLayoutSettingDict = windowsLayoutSetting [windowName].value
                            
            for settingName, val in singleWindowDict.iteritems():
                if str(settingName) in ['planePosition']:
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'int')
                    
                if str(settingName) in ['cameraClippingRange' , 'cameraFocalPoint' , 'cameraPosition' , 'cameraViewUp']:
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'floatlist')
                    
                elif str(settingName) in ['is3D']:    
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'bool')
                # elif str(settingName) in ['ArrowLength','MinRange','MaxRange']:    
                    # singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'float')                
                elif str(settingName) in ['winSize']:    
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'size')       
                elif str(settingName) in ['winPosition']:    
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),val,'point')                           
                elif str(settingName) in ['sceneName','sceneType','planeName','winType']:
                    singleWindowsLayoutSettingDict [str(settingName)] = Setting(str(settingName),str(val),'str')       
                    
        return windowsLayoutSetting        

 
    def toDictOfDictsParams(self):
        dictOfDictsParams = {}
        # print 'self.value=',self.value
        # sys.exit()
        
        for subDictName, subDict in self.value.iteritems():
            dictOfDictsParams [subDictName] = {}
            subDictParams = dictOfDictsParams [subDictName]
            
            for settingName, setting in subDict.value.iteritems():
                subDictParams [setting.name] = setting.value
            
        # print 'fieldParams=',fieldParams
        return dictOfDictsParams 
    
    
    def fieldParams2Setting(self, fieldParams):

        for key,val in fieldParams.iteritems():
            if type(val) == type (Setting(None,None,None)):
                #fieldParams is in the proper format
                return fieldParams
            else:
                # fieldParams needs to be converted to a proper format
                break

        fieldParamsSetting = {}
        # print 'fieldParams2Setting: fieldParams=',fieldParams
        for fieldName, singleFieldDict in fieldParams.iteritems():
        
            fieldParamsSetting [fieldName] = Setting(fieldName,{},'dict')

            singleFieldParamsSettingDict = fieldParamsSetting [fieldName].value
                            
            for settingName, val in singleFieldDict.iteritems():
                # print 'CONFIGURATION settingName=',settingName
                if str(settingName) in ['NumberOfLegendBoxes','NumberAccuracy','NumberOfContourLines']:
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'int')
                elif str(settingName) in ['MaxRangeFixed','LegendEnable','MinRangeFixed','ScaleArrowsOn',
                                          'FixedArrowColorOn','OverlayVectorsOn','ContoursOn','ShowPlotAxes']:
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'bool')
                elif str(settingName) in ['ArrowLength','MinRange','MaxRange']:    
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'float')                
                elif str(settingName) in ['ArrowColor']:    
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),QColor(val),'color')       
                elif str(settingName) in ['ScalarIsoValues']:
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),str(val),'str')       
                    


        
        return fieldParamsSetting        
        
    # def toFieldParams(self):
    #     fieldParams = {}
    #     # print 'self.value=',self.value
    #     # sys.exit()
    #
    #     for fieldName, singleFieldDict in self.value.iteritems():
    #         fieldParams [fieldName] = {}
    #         singleFieldParams = fieldParams [fieldName]
    #
    #         for settingName, setting in singleFieldDict.value.iteritems():
    #             singleFieldParams [setting.name] = setting.value
    #
    #     # print 'fieldParams=',fieldParams
    #     return fieldParams

    def normalizeSettingFormat(self):
        if self.name == 'TypeColorMap':
            # pass
            self.value = self.typeColorMap2Setting(self.value)
            
            
        elif self.name == 'FieldParams':
            # print 'normalizeSettingFormat self.value=',self.value
            self.value = self.fieldParams2Setting(self.value)
            
        elif self.name == 'WindowsLayout':
            self.value = self.windowsLayout2Setting(self.value)
            
                
    def toObject(self):
    
        if self.name == 'TypeColorMap':
            return self.toTypeColorMap()
            
        elif self.name == 'FieldParams':
            # return self.toFieldParams()
            return self.toDictOfDictsParams()
            
        elif self.name == 'WindowsLayout':
            return self.toDictOfDictsParams()        
            
        return self.value
                 
class CustomSettings(object):
    def __init__(self):
        # from collections import defaultdict
        # self.__setting = defaultdict(lambda : [None,None])
        self.__nameSettingDict = {} # {setting_name : setting}        
        self.__typeSettingDictDict = {} # { type : {setting_name:setting} }
        
    def getNameSettingDict(self): return self.__nameSettingDict
    
    def getSettingNameList(self): return self.__nameSettingDict.keys()
    
    def getTypeSettingDictDict(self): return self.__typeSettingDictDict
    
    def initializeSetting(self,_name, _value, _type=None):
        settingType =_type
        
            
        try:
            setting = self.__nameSettingDict [_name] 
            setting.value = _value
            settingType =  setting.type  
            
            setting.normalizeSettingFormat()
            
        except LookupError,e:
            setting = Setting(_name, _value, settingType)                   
            
            # setting = Setting(_name, _value, _type)
            
            settingType =  setting.type
            
            setting.normalizeSettingFormat()
            
            self.__nameSettingDict [_name] = setting
            
        try:
            # print '\n\n\n self.__typeSettingDictDict = ', self.__typeSettingDictDict             
            setting_dict = self.__typeSettingDictDict[settingType]
            # setting_dict = self.__typeSettingDictDict[_type]
            setting_dict [_name] = setting
        except LookupError,e:
            # self.__typeSettingDictDict[_type] = {_name:setting}
            self.__typeSettingDictDict[settingType] = {_name:setting}
    
    def setSetting(self,_name, _value, _type=None):
        settingType =_type
        
            
        try:
            setting = self.__nameSettingDict [_name] 
            setting.value = _value
            settingType =  setting.type  
            
            setting.normalizeSettingFormat()
            
        except LookupError,e:
            #check first default settings to get the type of the setting:
            # defaultSettingTemplate = defaultSettings().getSetting(_key)
            defaultSettingTemplate = defaultSettings().getSetting(_name)
            if defaultSettingTemplate:
                settingType = defaultSettingTemplate.type
            
            setting = Setting(_name, _value, settingType)                   
            
            # setting = Setting(_name, _value, _type)
            
            settingType =  setting.type
            
            setting.normalizeSettingFormat()
            
            self.__nameSettingDict [_name] = setting
            
        try:
            # print '\n\n\n self.__typeSettingDictDict = ', self.__typeSettingDictDict             
            setting_dict = self.__typeSettingDictDict[settingType]
            # setting_dict = self.__typeSettingDictDict[_type]
            setting_dict [_name] = setting
        except LookupError,e:
            # self.__typeSettingDictDict[_type] = {_name:setting}
            self.__typeSettingDictDict[settingType] = {_name:setting}
        
    def getSetting(self,_name):
        # type: (object) -> object
        # type: (object) -> object
        
        try:
            setting = self.__nameSettingDict[_name]
            return setting    
        except LookupError,e:
            return None
            
    def readFromXML(self,_fileName):
        import XMLUtils    
        xml2ObjConverter = XMLUtils.Xml2Obj()
        root_element=xml2ObjConverter.Parse(_fileName)
        settingsElemList=XMLUtils.CC3DXMLListPy(root_element.getElements("Settings"))
        
        readType2executeType={'int':'setting.XML2Int',
        'float':'setting.XML2Float',
        'str':'setting.XML2Str',
        'bool':'setting.XML2Bool',
        'strlist':'setting.XML2Strlist',
        'dict':'setting.XML2Dict',
        'color':'setting.XML2Color',
        'size':'setting.XML2Size',
        'point':'setting.XML2Point',
        'bytearray':'setting.XML2ByteArray'
        }        
            
        for elem in settingsElemList:
            type = elem.getAttribute("Type")
            # print type
                
            if type in readType2executeType.keys():
            # ['int','float','str','color','size','point']:
                elementList = XMLUtils.CC3DXMLListPy(elem.getElements(""))
                for el in elementList:
                
                    setting = Setting(el.name,el.cdata,type)
                    setting.fromXML(el)
                    

                    self.setSetting(setting.name,setting.value,setting.type)             
                    # if setting.name == 'FieldParams':
                        # print 'READ THIS FIELD PARAMS = ',setting.value

                    
            # print 'self.__typeSettingDictDict=',self.__typeSettingDictDict
            
            # print '\n\n\nself.__nameSettingDict=',self.__nameSettingDict
            
        # sys.exit()
            
    def saveAsXML(self, _fileName):
        print '_fileName=',_fileName
        import XMLUtils
        from XMLUtils import ElementCC3D
        import Version
        xml2ObjConverter = XMLUtils.Xml2Obj()
        plSetElem = ElementCC3D('PlayerSettings',{'version':Version.getVersionAsString()})
        # print '\n\n\nself.__typeSettingDictDict.keys() = ', self.__typeSettingDictDict.keys()
        # print '__typeSettingDictDict=',self.__typeSettingDictDict
        for typeName , settingDict in self.__typeSettingDictDict.iteritems():

            typeContainerElem = plSetElem.ElementCC3D( 'Settings', {'Type':typeName} )
            # print 'typeName=',typeName
            # if typeName =='FieldParams':
            #     print 'typeName=',typeName, ' settingDict=',settingDict

            for settingName, setting in sorted(settingDict.iteritems()):  # keys are sorted before outputting to XML
                # if settingName=='ShowPlotAxes':
                #
                #     try:
                #         print 'settingName=', settingName, ' setting=', setting, 'typeContainerElem=',typeName
                #     except:
                #         pass

                setting.toXML(typeContainerElem)

        fileFullPath = os.path.abspath(_fileName)
        plSetElem.CC3DXMLElement.saveXML(fileFullPath)
        # plSetElem.CC3DXMLElement.saveXML(fileFullPath+'.xml')

        
#the defaultSettings fcn will only be used intenally during development - it shold not be used int heproduction ode - the default settings shuld be read from the _settings.xml located in the COnfiguration directory of the player
def defaultSettings():

    defaultSettings = CustomSettings()
    
    # setsetting function
    # ss = defaultSettings.setSetting
    
    def ss (name,value,type):
        # defaultSettings.setSetting (name,Setting(name,value,type))
        # defaultSettings.setSetting (name,value,type)
        defaultSettings.initializeSetting (name,value,type)
    
    ss("TabIndex", 0, 'int')
    ss("RecentFile", '', 'str')
    ss('RecentSimulations',[],'strlist')
    ss('ScreenUpdateFrequency',10,'int')
    ss('ImageOutputOn',False,'bool')
    ss('SaveImageFrequency',100,'int')
    ss('Screenshot_X',600,'int')
    ss('Screenshot_Y',600,'int')
    ss('LatticeOutputOn',False,'bool')
    ss('SaveLatticeFrequency',100,'int')
    ss('GraphicsWinWidth',400,'int')
    ss('GraphicsWinHeight',400,'int')
    ss('UseInternalConsole',False,'bool')
    ss('ClosePlayerAfterSimulationDone',False,'bool')
    ss('ProjectLocation',os.path.expanduser('~'),'str')
    ss('OutputLocation',os.path.join(os.path.expanduser('~'),'CC3DWorkspace'),'str')
    ss('OutputToProjectOn',False,'bool')
    # ss('PreferencesFile',SETTINGS_FILE_NAME,'str') #probably do not need this one
    ss('NumberOfRecentSimulations',8,'int')
    ss('ContoursOn',False,'bool')
   
    
    typeColorMap = { 
        0:QColor(Qt.black),
        1:QColor(Qt.green),
        2:QColor(Qt.blue),
        3:QColor(Qt.red),
        4:QColor(Qt.darkYellow),
        5:QColor(Qt.lightGray),
        6:QColor(Qt.magenta),
        7:QColor(Qt.darkBlue),
        8:QColor(Qt.cyan),
        9:QColor(Qt.darkGreen),
        10:QColor(Qt.white)
        }    
    
    ss('TypeColorMap',typeColorMap,'dict')
    
    ss('BorderColor',QColor(Qt.yellow),'color')
    ss('ClusterBorderColor',QColor(Qt.blue),'color')
    ss('ContourColor',QColor(Qt.white),'color')
    ss('WindowColor',QColor(Qt.black),'color')
    
    ss('BrushColor',QColor(Qt.white),'color')
    ss('PenColor',QColor(Qt.black),'color')

    ss('WindowColorSameAsMedium',True,'bool')    
    ss('CellGlyphScaleByVolumeOn',False,'bool')    
    
    ss('CellGlyphScale',1.0,'float')    
    ss('CellGlyphThetaRes',2,'int')    
    ss('CellGlyphPhiRes',2,'int')            
    
    ss('PixelizedScalarField',False,'bool')    
    
    ss('FieldIndex',2,'int')                    
    ss('MinRange',0.0,'float')
    ss('MaxRange',1.0,'float')

    ss('MinRangeFixed',False,'bool')    
    ss('MaxRangeFixed',False,'bool')    
        
    ss('NumberOfLegendBoxes',2,'int')
    ss('NumberAccuracy',2,'int')
    ss('LegendEnable',False,'bool')    
    
    ss('ScalarIsoValues','','str')        
    ss('NumberOfContourLines',0,'int') 
    
    ss('ScaleArrowsOn',False,'bool')            
    ss('ArrowColor',QColor(Qt.white),'color')                
    ss('ArrowLength',0,'float')         
    
    ss('ArrowLength',0,'float')             
    
    ss('FixedArrowColorOn',False,'bool')             
    ss('OverlayVectorsOn',False,'bool')      

    
    ss('Types3DInvisible','0','str')             
    ss('BoundingBoxOn',True,'bool')                 
    ss('BoundingBoxColor',QColor(Qt.white),'color')                     
    ss('PlayerSizes',QByteArray(),'bytearray')
    
    ss('MainWindowSize',QSize(900, 650),'size')        
    ss('MainWindowPosition',QPoint(0,0),'point')                
        
    ss('Projection',0,'int')                        
    ss('CellsOn',True,'bool')                        
    ss('CellBordersOn',True,'bool')                        
    ss('ClusterBordersOn',False,'bool')                                
    ss('CellGlyphsOn',False,'bool')                                
    ss('FPPLinksOn',False,'bool')                                    
    ss('FPPLinksColorOn',False,'bool')                                    
    ss('ConcentrationLimitsOn',True,'bool')                                    
    ss('CC3DOutputOn',True,'bool')                                            
    ss('ZoomFactor',1.0,'float')                                                    
    ss('FieldParams',{},'dict')                                                            
        
        

    
    
    # print '\n\n\n defaultSettings dict=', defaultSettings.getNameSettingDict()
    return defaultSettings
        