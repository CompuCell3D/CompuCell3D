from PyQt4.QtGui import *
from PyQt4.QtCore import *

#from Messaging import stdMsg, dbgMsg,pd, errMsg, setDebugging
#setDebugging(1)

from os import environ,path
import os

#(ORGANIZATION, APPLICATION) = ("Biocomplexity", "PyQtPlayerNew")
(ORGANIZATION, APPLICATION) = ("Biocomplexity", "cc3d_default")
LATTICE_TYPES = {"Square":1,"Hexagonal":2}

maxNumberOfRecentFiles=5

MODULENAME = '------- player/Configuration/__init__.py: '

class Setting(object):
    # storedType2executeType = {'int':'int','float':'float','bool':'bool','str':'str','color':'QColor','size':'self.QSizeInit','point':'self.QPointInit','bytearray':'self.QByteArrayInit'}
    
    # storedType2StringConvertion = {'int':'str','float':'str','bool':'str','str':'str','dictsetting':'self.DictSettingToString','typecolormap':'self.TypeColorMapToString','color':'self.QColorToString','size':'self.QSizeToString','point':'self.QPointToString','bytearray':'self.QByteArrayToString'}
    # readType2executeType={'int':'int','float':'float','bool':'self.StringToBool','str':'str','color':'self.StringToQColor','size':'self.StringToQSize','point':'self.StringToQPoint','bytearray':'self.StringToQByteArray'}
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
        return self.__name+':['+str(self.__value)+','+str(self.__type)+']'
        
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
        # print 'Setting.storedType2XML[self.type]=',Setting.storedType2XML[self.type]
        eval(Setting.storedType2XML[self.type]+'(parentElement)')
        
    def fromXML(self,parentElement):
        # print 'Setting.storedType2XML[self.type]=',Setting.storedType2XML[self.type]
        eval(Setting.XML2StoredType[self.type]+'(parentElement)')
        
    def StringToQColor(self,_colorStr):        
        return QColor(_colorStr)
        
    def StringToQSize(self,_sizeStr):
        sizeList = _sizeStr.split(',')
        sizeListInt =  map (int, sizeList)
        
        return QSize(sizeListInt[0],sizeListInt[1])

    def StringToQPoint(self,_pointStr):
        pointList = _pointStr.split(',')
        pointListInt =  map (int, pointList)
        
        return QPoint(pointListInt[0],pointListInt[1])
        
    def StringToBool(self,_boolStr):
        if _boolStr == 'True':
            return True
            
        return False    
        
    # def StringToTypeColorMap(self,_typeColorMapStr):
        # tcmList = _typeColorMapStr.split(',')
        # typeColorMap = {}
        # for i in xrange(0,len(tcmList),2):
            # typeColorMap [int(tcmList[i])] =  Setting(str(tcmList[i]),QColor(str(tcmList[i+1])),'color' )
        # return typeColorMap
        
    def StringToDictSetting(self,_dictSettingStr):
        dsList = _dictSettingStr.split(',')
        print dsList
        dictSetting = {}
        for i in xrange(0,len(dsList),3):
            # print dsList[i]
            # print dsList[i+1]
            # print dsList[i+2]
            dictSetting [str(dsList[i])] =  eval (Setting.readType2executeType [str(dsList[i+2])] + '(' + dsList[i+1] + ')' )
        # print 'dictSetting=',dictSetting
        # sys.exit()    
        return dictSetting

        
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
    
        return str(_val.y())+','+str(_val.x())

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
        # return ','.join(map (str , _val))

    def DictSettingToString(self, _val):
        
        out_str = ''        
        for k,v in _val.iteritems():
            out_str += k + ',' + eval(Setting.storedType2StringConvertion [v.type] + '(v.value)' ) + ',' + v.type + ','
            
        return out_str[:-1]
    # def FieldParamsToString(self, _val):
    
        # return ','.join(map (str , _val))

        
    def toString(self):
        if self.type in Setting.storedType2StringConvertion.keys():
            # return self.QColorToString(self.value)
            return eval(Setting.storedType2StringConvertion[self.type]+'(self.value)')
        else:
            return self.value

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
            # print 'val=',val
            # print 'type val=',type(val)
            # print 'val.value=',val.value
            
            typeColorMap [int(key)] =  val.value
            
        # print 'typeColorMap=',typeColorMap    
        
        return typeColorMap

    def fieldParams2Setting(self, fieldParams):
    
        for key,val in fieldParams.iteritems():
            if type(val) == type (Setting(None,None,None)):
                #fieldParams is in the proper format
                return fieldParams
            else:
                #fieldParams needs to be converted to a proper format
                break    
                
        # print 'fieldParams=',fieldParams        
        # sys.exit()        
        
        fieldParamsSetting = {}
        
        for fieldName, singleFieldDict in fieldParams.iteritems():
        
            fieldParamsSetting [fieldName] = Setting(fieldName,{},'dict')
                        
            
            singleFieldParamsSettingDict = fieldParamsSetting [fieldName].value
            
            # singleFieldDictObject = None
            # if type(singleFieldDict) == type (Setting(None,None,None)):
                # singleFieldDictObject = singleFieldDict.value
            # else:
                # singleFieldDictObject = singleFieldDict
                
            for settingName, val in singleFieldDict.iteritems():
                if str(settingName) in ['NumberOfLegendBoxes','NumberAccuracy','NumberOfContourLines']:
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'int')
                elif str(settingName) in ['MaxRangeFixed','LegendEnable','MinRangeFixed','ScaleArrowsOn','FixedArrowColorOn','OverlayVectorsOn']:    
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'bool')
                elif str(settingName) in ['ArrowLength','MinRange','MaxRange']:    
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),val,'float')                
                elif str(settingName) in ['ArrowColor']:    
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),QColor(val),'color')       
                elif str(settingName) in ['ScalarIsoValues']:
                    singleFieldParamsSettingDict [str(settingName)] = Setting(str(settingName),str(val),'str')       
                    
                # if type(val) == type (Setting(None,None,None)):
                    # singleFieldParamsSetting
                # else:
                    # singleFieldParamsSetting [settingName] = Setting(str(key),val,'color')    
            
            # if type(val) == type (Setting(None,None,None)):
                # typeColorMapSetting [str(key)] = Setting(str(key),val.value,'color')    
            # else:
                # typeColorMapSetting [str(key)] = Setting(str(key),val,'color')    
            
        # print 'typeColorMapSetting=',typeColorMapSetting    
        print 'fieldParamsSetting=',fieldParamsSetting
        
        return fieldParamsSetting        
        
    def toFieldParams(self):
        fieldParams = {}
        # print 'self.value=',self.value
        # sys.exit()
        
        for fieldName, singleFieldDict in self.value.iteritems():
            fieldParams [fieldName] = {}
            singleFieldParams = fieldParams [fieldName]
            
            for settingName, setting in singleFieldDict.value.iteritems():
                singleFieldParams [setting.name] = setting.value
            
        # print 'fieldParams=',fieldParams
        # sys.exit()
        return fieldParams

    def normalizeSettingFormat(self):
        if self.name == 'TypeColorMap':
            # pass
            self.value = self.typeColorMap2Setting(self.value)
            
            
        elif self.name == 'FieldParams':
            self.value = self.fieldParams2Setting(self.value)
            
            
        
        
    def toObject(self):
    
        if self.name == 'TypeColorMap':
            return self.toTypeColorMap()
            
        elif self.name == 'FieldParams':
            return self.toFieldParams()
            
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
    
    def setSetting(self,_name, _value, _type=None):
        settingType =_type
            
        try:
            setting = self.__nameSettingDict [_name] 
            setting.value = _value
            settingType =  setting.type  
            
            setting.normalizeSettingFormat()
            
        except LookupError,e:
                    
            setting = Setting(_name, _value, _type)
            
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
        
        # readType2executeType={'int':'int','float':'float','str':'str','color':'self.StringToQColor','size':'el.StringToQSize','point':'el.StringToQPoint','bytearray':'el.StringToQByteArray'}
        # readType2executeType={'int':'int','float':'float','str':'str','bool':'setting.StringToBool','dict':'setting.XMLToDict','typecolormap':'setting.StringToTypeColorMap','color':'setting.StringToQColor','size':'setting.StringToQSize','point':'setting.StringToQPoint','bytearray':'setting.StringToQByteArray'}
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
            print type            
                
            if type in readType2executeType.keys():
            # ['int','float','str','color','size','point']:
                elementList = XMLUtils.CC3DXMLListPy(elem.getElements(""))
                for el in elementList:
                
                    setting = Setting(el.name,el.cdata,type)
                    setting.fromXML(el)
                    

                    self.setSetting(setting.name,setting.value,setting.type)             
                    if setting.name == 'FieldParams':
                        print 'READ THIS FIELD PARAMS = ',setting.value

                    
            print 'self.__typeSettingDictDict=',self.__typeSettingDictDict
            
            print '\n\n\nself.__nameSettingDict=',self.__nameSettingDict
            
        # sys.exit()
            
    def saveAsXML(self, _fileName):
        import XMLUtils
        from XMLUtils import ElementCC3D
        xml2ObjConverter = XMLUtils.Xml2Obj()
        plSetElem = ElementCC3D('PlayerSettings')    
        
        for typeName , settingDict in self.__typeSettingDictDict.iteritems():
        
            typeContainerElem = plSetElem.ElementCC3D( 'Settings', {'Type':typeName} )
            
            for settingName, setting in sorted(settingDict.iteritems()): # keys are sorted before outputting to XML
                setting.toXML(typeContainerElem)
                # if typeName in ['int','float','bool','color','str','bytearray','point','size','strlist','typecolormap','dict']:
                    # if settingName=='FieldParams':
                        # print 'SAVE XML FieldParams keys = ',setting.value.keys()
                        # print ' SAVE XMLfield params', setting
                    # setting.toXML(typeContainerElem)
     
        

        fileFullPath = os.path.abspath(_fileName)
        plSetElem.CC3DXMLElement.saveXML(fileFullPath)        
        
        # cs_local = CustomSettings()
        # cs_local.readFromXML(_fileName)
        # sys.exit()
        
    def extractSingleTypeSettings(self,_settingList ,_type ,_skipList=[]):
        for settingName in _settingList:
        
            val = Configuration.mySettings.value(settingName)
            valToSave=None
            typeToSave = _type
            
            if val.isValid():
                if _type == 'int':
                    valToSave ,ok = val.toInt() # toInt returns tuple: first = integer; second = flag                            
                elif _type == 'bool':
                    valToSave = val.toBool()
                elif _type == 'str':
                    valToSave = val.toString()
                elif _type == 'float':
                    valToSave ,ok = val.toDouble() # toDouble returns tuple: first = double; second = flag                        
                elif _type == 'color':
                    print 'color val=',val                    
                    valToSave = QColor(val.toString())   
                elif _type in ['size']:
                    valToSave = val.toSize()       
                elif _type in ['point']:
                    valToSave = val.toPoint()
                elif _type == 'bytearray':
                    valToSave  = val.toByteArray()
                elif _type == 'typecolormap': 
                    intColorList = val.toStringList()                
                    typeToSave = 'dict'
                    valToSave={}
                    k = 0 
                    for i in range(intColorList.count()/2):
                    
                        key, ok  = intColorList[k].toInt()
                        k  += 1
                        value   = intColorList[k]
                        k  += 1                
                        if ok:
                            valToSave[str(key)] = Setting(str(key), QColor(value),'color')
                
                elif _type == 'strlist':
                    
                    strList = val.toStringList()                
                                    
                    valToSave=[]
                    for i in range(strList.count()):
                        valToSave.append(str(strList[i]))

                
                elif _type == 'fieldparams':
                    
                    typeToSave = 'dict'
                    
                    if settingName == "FieldParams":
                        fieldParamMap = val.toMap()
                                        
                        valToSave={}
                        print 'fieldParamMap=',fieldParamMap
                        
                        for fieldNameQStr, fieldParamMapQVar in fieldParamMap.iteritems():
                        
                            
                            fieldName = str(fieldNameQStr)
                            
                            valToSave[fieldName]=Setting(fieldName,{},'dict')
                            valToSaveFieldParamDict = valToSave[fieldName].value
                            print 'fieldName=',fieldName
                            
                            # # valToSave[fieldName]={}
                            # # valToSaveFieldParamDict = valToSave[fieldName]
                            
                            fieldParamMap = fieldParamMapQVar.toMap()
                            print 'fieldParamMap=',fieldParamMap
                            
                            for paramNameQStr, paramValue in fieldParamMap.iteritems():
                                paramName = str(paramNameQStr)
                                val = None
                                type = None
                                if paramName in ['OverlayVectorsOn','ScaleArrowsOn','FixedArrowColorOn','MaxRangeFixed','MinRangeFixed','LegendEnable']:
                                    val = paramValue.toBool()
                                    type = 'bool'
                                elif paramName in ['ArrowLength','NumberOfContourLines','NumberAccuracy','NumberOfLegendBoxes','Length']:
                                    val,ok = paramValue.toInt()
                                    type = 'int'
                                elif paramName in ['MaxRange','MinRange']:
                                    val,ok = paramValue.toDouble()
                                    type = 'float'
                                elif paramName in ['ScalarIsoValues']:
                                    val= paramValue.toString()
                                    type = 'str'
                                elif paramName in ['ArrowColor']:
                                    val = QColor(paramValue.toString())
                                    type = 'color'
                                else:
                                    val = str(paramValue)
                                    type = 'str'
                                
                                valToSaveFieldParamDict[paramName] = Setting(paramName,val,type)
                                print 'paramName=',paramName,' paramValue=',paramValue
                        # sys.exit()
                        print 'valToSave=',valToSave        
                        # sys.exit()
                self.setSetting(settingName,valToSave,typeToSave)                             
                                
                # elif _type == 'fieldparams':
                    
                    # if settingName == "FieldParams":
                        # fieldParamMap = val.toMap()
                                        
                        # valToSave={}
                        # print 'fieldParamMap=',fieldParamMap
                        
                        # for fieldNameQStr, fieldParamMapQVar in fieldParamMap.iteritems():
                        
                            # fieldName = str(fieldNameQStr)
                            # valToSave[fieldName]={}
                            # valToSaveFieldParamDict = valToSave[fieldName]
                            
                            # fieldParamMap = fieldParamMapQVar.toMap()
                            # print 'fieldParamMap=',fieldParamMap
                            # for paramNameQStr, paramValue in fieldParamMap.iteritems():
                                # paramName = str(paramNameQStr)
                                # val = None
                                # type = None
                                # if paramName in ['OverlayVectorsOn','ScaleArrowsOn','FixedArrowColorOn','MaxRangeFixed','MinRangeFixed','LegendEnable']:
                                    # val = paramValue.toBool()
                                    # type = 'bool'
                                # elif paramName in ['ArrowLength','NumberOfContourLines','NumberAccuracy','NumberOfLegendBoxes','Length']:
                                    # val,ok = paramValue.toInt()
                                    # type = 'int'
                                # elif paramName in ['MaxRange','MinRange']:
                                    # val,ok = paramValue.toDouble()
                                    # type = 'float'
                                # elif paramName in ['ScalarIsoValues']:
                                    # val= paramValue.toString()
                                    # type = 'str'
                                # elif paramName in ['ArrowColor']:
                                    # val = QColor(paramValue.toString())
                                    # type = 'color'
                                # else:
                                    # val = str(paramValue)
                                    # type = 'str'
                                
                                # valToSaveFieldParamDict[paramName] = Setting(paramName,val,type)
                                # print 'paramName=',paramName,' paramValue=',paramValue
                        # print 'valToSave=',valToSave        
                        # sys.exit()    

                                
 
        
    
    def extractCustomSettingsFromGlobals(self):
                
        # extracting int type values
         self.extractSingleTypeSettings( _settingList = Configuration.paramTypeInt,_type = 'int',_skipList=[])
        # extracting bool type values
         self.extractSingleTypeSettings( _settingList = Configuration.paramTypeBool,_type = 'bool',_skipList=[])
        # extracting str type values
         self.extractSingleTypeSettings( _settingList = Configuration.paramTypeString,_type = 'str',_skipList=[])    
        # extracting float type values
         self.extractSingleTypeSettings( _settingList = Configuration.paramTypeDouble,_type = 'float',_skipList=[])    
        # extracting color type values
         self.extractSingleTypeSettings( _settingList = Configuration.paramTypeColor,_type = 'color',_skipList=[])
        # extracting string list type values
         self.extractSingleTypeSettings( _settingList = ['RecentSimulations'],_type = 'strlist',_skipList=[])
        # extracting string list type values for color map
         self.extractSingleTypeSettings( _settingList = ['TypeColorMap'] , _type = 'typecolormap',_skipList=[])
        # extracting qsize type values 
         self.extractSingleTypeSettings( _settingList = ["MainWindowSize"] , _type = 'size',_skipList=[])
        # extracting qpoint type values 
         self.extractSingleTypeSettings( _settingList = ["MainWindowPosition"] , _type = 'point',_skipList=[])        
        # extracting fieldparams type values 
         self.extractSingleTypeSettings( _settingList = ["FieldParams"] , _type = 'fieldparams',_skipList=[])
        # extracting bytearray type values 
         self.extractSingleTypeSettings( _settingList = ["PlayerSizes"] , _type = 'bytearray',_skipList=[])

def defaultSettings():

    defaultSettings = CustomSettings()
    
    # setsetting function
    ss = defaultSettings.setSetting
    
    def ss (name,value,type):
        # defaultSettings.setSetting (name,Setting(name,value,type))
        defaultSettings.setSetting (name,value,type)
    
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
    ss('ProjectLocation',os.path.join(environ['PREFIX_CC3D'],'Demos'),'str')
    ss('OutputLocation',os.path.join(os.path.expanduser('~'),'CC3DWorkspace'),'str')
    ss('OutputToProjectOn',False,'bool')
    # ss('PreferencesFile','_setting.xml','str') #probably do not need this one
    ss('NumberOfRecentSimulations',8,'int')
    
   
    
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

def loadGlobalSettings():
    global_setting_dir = os.path.abspath(os.path.join(os.path.expanduser('~'),'.compucell3d'))
    global_setting_path = os.path.abspath(os.path.join(global_setting_dir,'_settings.xml')) # abspath normalizes path
    print 'LOOKING FOR global_setting_path=',global_setting_path
    
    #create global settings  directory inside user home directory
    if not os.path.isdir(global_setting_dir):
        try:
            os.makedirs(global_setting_dir)
    
        except:
            print 'Cenfiguration: COuld not make directory: ',global_setting_dir, ' to store global settings. Please make sure that you have appropriate write permissions'
            import sys
            sys.exit()
            
    if os.path.isfile(global_setting_path):
        import XMLUtils
        xml2ObjConverter = XMLUtils.Xml2Obj()

        fileFullPath = os.path.abspath(global_setting_path)
        gs = CustomSettings()        
        gs.readFromXML(global_setting_path)
        print 'gs=',gs
        return gs,global_setting_path
        
    else:    
        print ' NOT FOUND ',global_setting_path    
        globalSettings = defaultSettings()
        globalSettings.saveAsXML(global_setting_path)        
        
        return globalSettings , global_setting_path   
                
class Configuration():
        
    myGlobalSettings,myGlobalSettingsPath = loadGlobalSettings()         
    
    myCustomSettings = None # this is an object that stores settings for custom settings i.e. ones which are associated with individual cc3d projects
    myCustomSettingsPath = ''
    
    globalOnlySettings = ['RecentSimulations','NumberOfRecentSimulations']
    
    activeFieldNamesList = []

def getSettingNameList(): return Configuration.myGlobalSettings.getSettingNameList()

def addItemToRecentSimulations(item):

    currentStrlist = getSetting('RecentSimulations')
    maxLength = getSetting('NumberOfRecentSimulations')
       
    if maxLength <0:
        currentStrlist.insert(0,item)
    else:
        currentStrlist.insert(0,item)
        # print  'len(currentStrlist)=',len(currentStrlist),' maxLength=',maxLength   
        if len(currentStrlist) > maxLength:
            currentStrlist = currentStrlist[: - ( len(currentStrlist)-maxLength ) ] 
        print 'maxLength=',maxLength    
        print 'currentStrlist=',currentStrlist
        
    #eliminating duplicates        
    seen = set()
    seen_add = seen.add
    currentStrlist = [ x for x in currentStrlist if not (x in seen or seen_add(x))]        
    
    # setSetting('RecentSimulations',currentStrlist)
    
    print 'currentStrlist=',currentStrlist
    
    # sys.exit()
        
        
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
            
    print 'cleanedFieldParams.keys() = ', cleanedFieldParams.keys()   
    # import time
    # time.sleep(2)
    
    print 'cleanedFieldParams =', cleanedFieldParams
    
    setSetting('FieldParams',cleanedFieldParams)
    # sys.exit()
    
def writeSettings (settingsObj,path):
    if settingsObj:
        settingsObj.saveAsXML(path)

def writeAllSettings():

    print 'Configuration.myGlobalSettings.typeNames = ', Configuration.myGlobalSettings.getTypeSettingDictDict().keys()
    print 'Configuration.myGlobalSettings. = ', Configuration.myGlobalSettings.getTypeSettingDictDict()
    
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
        
# def writeSettings():
        
    # Configuration.myGlobalSettings.saveAsXML(Configuration.myGlobalSettingsPath)
    # # print 'SAVED Configuration.myGlobalSettingsPath=',Configuration.myGlobalSettingsPath
    
    
    # if Configuration.myCustomSettingsPath:
        # Configuration.myCustomSettings.saveAsXML(Configuration.myCustomSettingsPath)
    

def readCustomFile(fileName):
    
    
    import XMLUtils
    xml2ObjConverter = XMLUtils.Xml2Obj()

    fileFullPath = os.path.abspath(fileName)
    cs = CustomSettings()        
    cs.readFromXML(fileFullPath)
    # # # FIX HERE
    Configuration.myCustomSettings = cs
    Configuration.myCustomSettingsPath = fileName
    
    # root_element = xml2ObjConverter.Parse(fileFullPath) # this is simulation element    
    # if root_element.getFirstElement("ScreenUpdateFrequency"):
        # ScreenUpdateFrequency = int(root_element.getFirstElement("ScreenUpdateFrequency").getText()            )
        # print 'READ ScreenUpdateFrequency=',ScreenUpdateFrequency

    # from CustomSettings import CustomSettings            
    # Configuration.myCustomSettings = CustomSettings()
    # Configuration.myCustomSettings.setValue('ScreenUpdateFrequency',ScreenUpdateFrequency,'int')


# def writeCustomFile(fileName):

    # print 'filename=',fileName
    
    # import XMLUtils
    # from XMLUtils import ElementCC3D
    # settings=ElementCC3D("Settings")
    
    
    
    # Configuration.myCustomSettings
    
    # cs =None
    # if Configuration.myCustomSettings:
        # #if Configuration.myCustomSettings is initialized (e.g. read from XML) we will save existing CustomSetting object
        # cs = Configuration.myCustomSettings
    # else:        
        # #else (e.g. simulation was opened witout custom settings), we will create new CustomSettings object and extract to it global settings
        # cs = CustomSettings()        
        # cs.extractCustomSettingsFromGlobals()
        # # # # FIX HERE
        # Configuration.myCustomSettings = cs
        
    # cs.saveAsXML(fileName)
    
    
# def writeCustomFile(fileName):

    # print 'filename=',fileName
    
    # import XMLUtils
    # from XMLUtils import ElementCC3D
    # settings=ElementCC3D("Settings")
    
    
    # cs =None
    # if Configuration.myCustomSettings:
        # #if Configuration.myCustomSettings is initialized (e.g. read from XML) we will save existing CustomSetting object
        # cs = Configuration.myCustomSettings
    # else:        
        # #else (e.g. simulation was opened witout custom settings), we will create new CustomSettings object and extract to it global settings
        # cs = CustomSettings()        
        # cs.extractCustomSettingsFromGlobals()
        # # # # FIX HERE
        # Configuration.myCustomSettings = cs
        
    # cs.saveAsXML(fileName)
    
    
def setPrefsFile(fname):
    print
    print MODULENAME,'------------  setPrefsFile:  fname=',fname,'\n'
    Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, fname)

def getVersion():
    import Version
    return Version.getVersionAsString()
    
def getSimFieldsParams():

    fieldParams = Configuration.simFieldsParams

    if len(Configuration.simFieldsParams) == 0:

        fieldParams = getSetting('FieldParams')
        Configuration.simFieldsParams = fieldParams 

    return fieldParams

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
            # paramsDict = {}

            # paramsDict["MinRange"] = getSetting("MinRange")
            # paramsDict["MinRangeFixed"] = getSetting("MinRangeFixed")
            # paramsDict["MaxRange"] = getSetting("MaxRange")
            # paramsDict["MaxRangeFixed"] = getSetting("MaxRangeFixed")
            
            # paramsDict["NumberOfLegendBoxes"] = getSetting("NumberOfLegendBoxes")
            # paramsDict["NumberAccuracy"] = getSetting("NumberAccuracy")
            # paramsDict["LegendEnable"] = getSetting("LegendEnable")
        
            # paramsDict["NumberOfContourLines"] = getSetting("NumberOfContourLines")

        
            # paramsDict["ScaleArrowsOn"] = getSetting("ScaleArrowsOn")
            # color = getSetting("ArrowColor")

            # paramsDict["ArrowColor"] = color

            # paramsDict["ArrowLength"] = getSetting("ArrowLength")
            # paramsDict["FixedArrowColorOn"] = getSetting("FixedArrowColorOn")
            # paramsDict["OverlayVectorsOn"] = getSetting("OverlayVectorsOn")
            # paramsDict["ScalarIsoValues"] = getSetting("ScalarIsoValues")

            # fieldParams[field] = paramsDict
            

    Configuration.simFieldsParams = fieldParams
    print 'initFieldsParams fieldParams = ',fieldParams
    setSetting('FieldParams',fieldParams )
    
def updateFieldsParams(fieldName,fieldDict):
    
    fieldParamsDict = getSetting("FieldParams")
    Configuration.simFieldsParams = fieldParamsDict
    # import time
    # time.sleep(1)
    
    # if not isinstance(fieldName,str):

    fieldName = str(fieldName)
    
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    
    # import time
    # time.sleep(1)
    # print 'update Field Params FieldParams = ',Configuration.simFieldsParams
    
    setSetting('FieldParams',Configuration.simFieldsParams)
    
    
        
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not

    
    # # # # import time
    # # # # time.sleep(1)
    # # # # print 'update Field Params FieldParams = ',Configuration.simFieldsParams
    
    # # # setSetting('FieldParams',Configuration.simFieldsParams)
    
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
    print 'maxLength=',maxLength    
    print 'currentStrlist=',currentStrlist
        
    
    # setSetting('RecentSimulations',currentStrlist)
    # val = currentStrlist
    print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n currentStrlist=',currentStrlist       
    # import time
    # time.sleep(2)
    
    return currentStrlist
    
    
def getSetting(_key, fieldName=None):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        
    print '_key=',_key
    settingStorage = None    
    
    if Configuration.myCustomSettings:
        settingStorage = Configuration.myCustomSettings
    else:
        settingStorage = Configuration.myGlobalSettings
        
    if _key in Configuration.globalOnlySettings: # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
        settingStorage = Configuration.myGlobalSettings
    
    val = settingStorage.getSetting(_key)  
    if val:
        return val.toObject()
        
    return None
        
def setSetting(_key,_value):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        
    print 'SETTING _key=',_key
        
    val = _value
    if _key == 'PlayerSizes':
        print 'player_sizes = ',_value
        # sys.exit()
    
    if _key == 'RecentSimulations' : # need to find better solution for that... 
        simName = _value
        val  = getRecentSimulationsIncludingNewItem(simName) # val is the value that is stored int hte settings
    
    
    if _key in Configuration.globalOnlySettings: # some settings are stored in the global settings e.g. number of recent simualtions or recent simulations list
        Configuration.myGlobalSettings.setSetting(_key,val)  
    else:    
        Configuration.myGlobalSettings.setSetting(_key,val)  # we always save everythign to the global settings
        if Configuration.myCustomSettings:
            Configuration.myCustomSettings.setSetting(_key,val)
    
    
    # Configuration.myGlobalSettings.setSetting(_key,_value)  
    # if Configuration.myCustomSettings:
        # Configuration.myCustomSettings.setSetting(_key,_value)
    
# def getPlayerParams():

    # playerParamsDict = {}
    # for key in Configuration.defaultConfigs.keys():
        # if key not in ["PlayerSizes"]:
            # playerParamsDict[key] = getSetting(key)


    # return playerParamsDict
    
    
def getSetting1(_key, fieldName=None):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        
    print '_key=',_key
    if fieldName:
        if fieldName == 'Cell_Field':  # if there are no fields defined, but just the Cell_Field, return default Pref (hard-coded -> BAD)
            return getSetting(_key)
        

        fieldsDict = getSimFieldsParams()
        
        try:
            paramsDict = fieldsDict[fieldName]
        except LookupError,e:
            return getSetting(_key) # returning default value stored in the setting for the field
            
        if _key == 'ArrowColor':  
            sys.exit()
            val = paramsDict[_key]

            r, g, b = (1,0,0)
            if isinstance(val,str) and val[0]=='#':  #   hex value, e.g.  '#ff0000   (i.e. #rrggbb)

                r, g, b = val[1:3], val[3:5], val[5:]
                r, g, b = [int(n, 16)/255. for n in (r, g, b)]   # return normalized in [0,1] for VTK
            else:

                r= val.red()/255.
                g= val.green()/255.
                b= val.blue()/255.
                
            return (r,g,b) 
        else:
            if _key not in paramsDict.keys():
                print MODULENAME, ' ------------>  WARNING:  getSetting(): _key not in paramsDict',_key,paramsDict
                return 0
            val = paramsDict[_key]

            return val 
    # elif _key in ['ScreenUpdateFrequency']:
        # # print 'Trying to open ScreenUpdateFrequency'
        # # print 'Configuration.myCustomSettings=',Configuration.myCustomSettings
        # if Configuration.myCustomSettings:
            # val = Configuration.myCustomSettings.getSetting('ScreenUpdateFrequency')  
            # # print 'val=',val.value
            
            # # value = int(val.value)
            # # return val.value
            # # print 'val=',val.toObject(),' type=',type(val.toObject())                
            # return val.toObject()
        # else:
            # val = Configuration.mySettings.value(_key)
            # if val.isValid():
                # return val.toInt()[0] # toInt returns tuple: first = integer; second = flag
            # else:
                # return Configuration.defaultConfigs[_key]
            
    elif _key in Configuration.paramTypeBool:
        # print 'got key=',_key
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            # v = val.toObject()
            # print 'Bool setting:',_key,' val=',v,' type=',type(v) 
            if val:
                return val.toObject()
            

    
        val = Configuration.mySettings.value(_key)

        if val.isValid():
            return val.toBool()
        else:
            return Configuration.defaultConfigs[_key]
    
    elif _key in Configuration.paramTypeString:
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)                  
            if val:
                # print 'String setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())  
                return val.toObject()
            
    
    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            return val.toString()
        else:
            return Configuration.defaultConfigs[_key]
    
    elif _key in Configuration.paramTypeInt:   # ["ScreenUpdateFrequency","SaveImageFrequency"]:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            # print 'Int setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
            if val:
                return val.toObject()

    
    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            return val.toInt()[0] # toInt returns tuple: first = integer; second = flag
        else:
            return Configuration.defaultConfigs[_key]
        
    elif _key in Configuration.paramTypeDouble:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            # print 'Float setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
            if val:
                return val.toObject()

    
    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            return val.toDouble()[0]
        else:
            return Configuration.defaultConfigs[_key]
        
    elif _key in Configuration.paramTypeColor:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            # print 'Color setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
            if val:
                return val.toObject()

    
    
        val = Configuration.mySettings.value(_key)

        if val.isValid():
            color = QColor(val.toString())
            return color
        else:
            color = Configuration.defaultConfigs[_key]

            return color
    
    elif _key in ["RecentSimulations"]:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
                            
            if val:
                # print 'RecentSimulations setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
                return val.toObject()
    
    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            recentSimulationsList = val.toStringList()
            recentSimulations=[]
            for i in range(recentSimulationsList.count()):
                recentSimulations.append(str(recentSimulationsList[i]))
            return recentSimulations
        else:
            return Configuration.defaultConfigs[_key]
    elif _key in ["FieldParams"]:
    
        print 'will fetch FIELD PARAMS'
        import time
        # time.sleep(1)
        
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)                  
            
            if val:
                # print 'FieldParams setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
                
                print 'from XML fieldParams keys= ',val.toObject().keys(),' val ',val
                
                return val.toObject()
                
        print 'FETCHING FIELD PARAMS Q SETTINGS'
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            fieldDict = val.toMap()

            fieldDictNew = {}
            knt = 0
            for key in fieldDict.keys():

                fieldDictNew[str(key)] = {}
                val = fieldDict.values()[knt]
                
                dict2 = val.toMap()

                dictVals = {}
                for key2 in dict2.keys():

                    if str(key2)[-2:] == 'On':  dictVals[str(key2)] = dict2[key2].toBool()
                    elif str(key2)[-5:] == 'Fixed':  dictVals[str(key2)] = dict2[key2].toBool()
                    elif str(key2) == 'LegendEnable':  dictVals[str(key2)] = dict2[key2].toBool()
                    elif str(key2)[:6] == 'Number':  dictVals[str(key2)] = dict2[key2].toInt()[0]  # e.g. toInt() -> (3,True)
                    elif str(key2)[-6:] == 'Length':  dictVals[str(key2)] = dict2[key2].toInt()[0]
                    elif str(key2)[-5:] == 'Range':  dictVals[str(key2)] = dict2[key2].toFloat()[0]
                    elif str(key2)[-4:] == 'List':  dictVals[str(key2)] = dict2[key2].toString()[0]
                    elif str(key2) == 'ScalarIsoValues':  dictVals[str(key2)] = dict2[key2].toString()
                    elif str(key2) == 'ArrowColor':  

                        dictVals[str(key2)] = str(dict2[key2].toString())
                        mycolor = QColor(dict2[key2].toString())

                    else:  dictVals[str(key2)] = dict2[key2]

                fieldDictNew[str(key)] = dictVals
                knt += 1
                
            print 'REGULAR fieldDictNew=',fieldDictNew    
            checkDict = fieldDictNew[fieldDictNew.keys()[0]]
            checkPhrase = 'OverlayVectorsOn'
            if checkPhrase in checkDict.keys():
                print 'FOUND '+checkPhrase
            else:
                print 'NOT FOUND '+checkPhrase
            # sys.exit()
            return fieldDictNew
        else:
            fieldDict = Configuration.defaultConfigs[_key]
            
            # rwh: need to call initFieldsParams
            return fieldDict
    elif _key in ["MainWindowSize","InitialSize"]: # QSize values

        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            
            # print 'QSize setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
            if val:
                return val.toObject()

    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            return val.toSize() 
        else:
            return Configuration.defaultConfigs[_key]                             

    elif _key in ["MainWindowPosition","InitialPosition"]: # QPoint values
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            
            # print 'QPoint setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
            if val:
                return val.toObject()

    
    
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            pval = val.toPoint()

            return val.toPoint() 
        else:
            return Configuration.defaultConfigs[_key]
        
    elif _key in ["PlayerSizes"]:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            
            if val:
                return val.toObject()

            # v = val.toObject()
            # print 'QByteArray setting:',_key,' val=',v,' type=',type(v)                
            # # return val.toObject()
            # for i in range(v.count()):
                # # print ord(ba[i]),
                # print 'v[',i,']=',ord(v[i])
    
            
        val = Configuration.mySettings.value(_key)
        if val.isValid():
            ba = val.toByteArray() 
            # print 'ba :'
            # for i in range(ba.count()):
                # # print ord(ba[i]),
                # print 'ba[',i,']=',ord(ba[i])
                
            return val.toByteArray() 
        else:
            return Configuration.defaultConfigs[_key]
        
    elif _key in ["TypeColorMap"]:
    
        if Configuration.myCustomSettings:
            val = Configuration.myCustomSettings.getSetting(_key)  
            print val
            
            if val:
                # print 'TypeColorMap setting:',_key,' val=',val.toObject(),' type=',type(val.toObject())                
                return val.toObject()

    

        colorMapStr = Configuration.mySettings.value(_key)
        

        if colorMapStr.isValid():
            colorList = colorMapStr.toStringList()


            if colorList.count() == 0:

                colorMapPy = Configuration.defaultConfigs["TypeColorMap"]
                colorList = QStringList()
                
                for _key in colorMapPy.keys():
                    colorList.append(str(_key))
                    colorList.append(colorMapPy[_key].name())

            import sys         
                
            # Do color dictionary                
            colorDict = {}
            k = 0
            for i in range(colorList.count()/2):
                key, ok  = colorList[k].toInt()
                k       += 1
                value   = colorList[k]
                k       += 1
                if ok:
                    colorDict[key]  = QColor(value)
                    
            # print 'colorDict=',colorDict
            
            
            return colorDict
        else:

            return Configuration.defaultConfigs["TypeColorMap"]

    else:
        print MODULENAME,' getSetting(), bogus key =',_key
        raise # exception

def addNewSimulation(recentSimulationsList,value):

    if str(value)=="":
        return False
    elementExists=False
    idxFound = -1
    for idx in range(recentSimulationsList.count()):
    
        if str(recentSimulationsList[idx])==value:
            elementExists=True
            idxFound=idx
            break
    if not elementExists:    

        recentSimulationsList.prepend(str(value))
        return True
    else:
        # moving existing item to the beginning of the list
        fileNameTmp = recentSimulationsList[idxFound]

        recentSimulationsList.removeAt(idxFound)
        recentSimulationsList.prepend(fileNameTmp)

        return False
                
def setSetting1(_key,_value):  # rf. ConfigurationDialog.py, updatePreferences()
    
    if _key in Configuration.paramTypeBool:            
        
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)
                
        Configuration.mySettings.setValue(_key,QVariant(_value))
        
    elif _key in Configuration.paramTypeInt:
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)  
            
    
        Configuration.mySettings.setValue(_key,_value)
        
    elif _key in Configuration.paramTypeDouble:
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)
    
    
        Configuration.mySettings.setValue(_key,QVariant(_value))
        
    elif _key in Configuration.paramTypeString:
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)

        Configuration.mySettings.setValue(_key,_value)
                
    elif _key in Configuration.paramTypeColor:
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)

        Configuration.mySettings.setValue(_key,_value)
    
    elif _key in ["RecentSimulations"]:
    

        recentSimulationsVariant = Configuration.mySettings.value("RecentSimulations")
        if recentSimulationsVariant.isValid():
            recentSimulationsList = recentSimulationsVariant.toStringList()

            maxNumberOfRecentFiles = getSetting("NumberOfRecentSimulations")
            
            if recentSimulationsList.count() > maxNumberOfRecentFiles: 
                
                removeNumber=recentSimulationsList.count()-maxNumberOfRecentFiles
                
                for i in xrange(removeNumber):
                    recentSimulationsList.removeAt(recentSimulationsList.count()-1)
            
            if recentSimulationsList.count() >= maxNumberOfRecentFiles:    
                addingSuccessful = addNewSimulation(recentSimulationsList,_value)

                if addingSuccessful:
                    recentSimulationsList.removeAt(recentSimulationsList.count()-1)
            else:

                addingSuccessful = addNewSimulation(recentSimulationsList,_value)
                
            Configuration.mySettings.setValue("RecentSimulations", QVariant(recentSimulationsList))  # each time we set a list of recent files we have to update variant variable corresponding to this setting to ensure that recent file list is up to date in the GUI                
                        

        else:
#                print "       recentSimulationsVariant is NOT valid:  _key,_value=",_key,_value
            recentSimulationsList = QStringList()
#                recentSimulationsList.prepend(QString(_value))
            Configuration.mySettings.setValue("RecentSimulations", QVariant(recentSimulationsList))
            
            
        if Configuration.myCustomSettings:
            # print 'storing STRLIST',_key,' ',_value
            # print 'type = ',type(_value)
            # import time
            # time.sleep(5)
            
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,recentSimulationsList)
                
            # print 'before RecentSimulations setValue'    

            # print 'after RecentSimulations setValue'    
            
        
    # # # # string
    # # # elif _key in ["BaseFontName","BaseFontSize"]:
        # # # Configuration.mySettings.setValue(_key,QVariant(_value))
    
    # QSize, QPoint,QStringList , QString
    # elif _key in ["InitialSize","InitialPosition","KeyboardShortcuts"]:
    elif _key in ["InitialSize","InitialPosition",]:
    
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)
                
        Configuration.mySettings.setValue(_key,QVariant(_value))
        
    elif _key in ["PlayerSizes","MainWindowSize","MainWindowPosition"]:
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)
                
        Configuration.mySettings.setValue(_key, QVariant(_value))
        
    elif _key in ["FieldParams"]:

        if Configuration.myCustomSettings:
            print '\n\n\n storing ',_key
            if type(_value) == dict:
                print 'STORED FIELD PARAMS keys = ',_value.keys(), '  field params',_value
            print 'type = ',type(_value)
            # import time
            # time.sleep(2)
            
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                print '\n\n\n SET SETTING FIELD PARAMS = ',_value
                Configuration.myCustomSettings.setSetting(_key,_value)
                
        if isinstance(_value,dict):

            Configuration.mySettings.setValue(_key, QVariant(_value))
        else:  # this block gets executed, it seems
            valDict = _value.toMap()
                    
            Configuration.mySettings.setValue(_key, QVariant(valDict))
        
            
        
    elif _key in ["TypeColorMap"]:
    
        if Configuration.myCustomSettings:
            # print 'storing ',_key,' ',_value
            # print 'type = ',type(_value)
            if type(_value) == type(QVariant()):
                pass
                # print 'got QVARIANT'
            else:    
                Configuration.myCustomSettings.setSetting(_key,_value)    

        penColorList = QStringList()
#            print '---  Config-/__init__.py: penColorList =',penColorList
        
        if type(_value) == dict:
#                print 'yes, _value is a dict'
#                print '---  Config-/__init__.py: setSetting: _key=TypeColorMap: len(_value) =',len(_value)
            for i in range(len(_value)):
                keys = _value.keys()
                penColorList.append(str(keys[i]))
                penColorList.append(str(_value[keys[i]].name()))

                
            Configuration.mySettings.setValue(_key, QVariant(penColorList))

        # rwh: I confess I'm confused, but it seems this block is not even needed
    
    else:

        print MODULENAME,"Wrong format of configuration option:" + str(_key) + ":" + str(_value)
            

def getPlayerParams1():

    playerParamsDict = {}
    for key in Configuration.defaultConfigs.keys():
        if key not in ["PlayerSizes"]:
            playerParamsDict[key] = getSetting(key)


    return playerParamsDict

def syncPreferences():   # this function invoked when we close the Prefs dialog with the "OK" button
    pass
    # for key in Configuration.defaultConfigs.keys():
        # val = Configuration.mySettings.value(key)

        # if val.isValid():  # if setting exists (is valid) in the .plist
            # if not key == 'RecentSimulations':
                # setSetting(key,val)
        # else:
            # print 'setting recent simulations'
            # setSetting(key,Configuration.defaultConfigs[key])

    
def syncPreferences1():   # this function invoked when we close the Prefs dialog with the "OK" button
    for key in Configuration.defaultConfigs.keys():
        val = Configuration.mySettings.value(key)

        if val.isValid():  # if setting exists (is valid) in the .plist
            if not key == 'RecentSimulations':
                setSetting(key,val)
        else:
            print 'setting recent simulations'
            setSetting(key,Configuration.defaultConfigs[key])
            
        

# class Configuration1():

        # #default settings
        # defaultConfigs={}
        
        # simFieldsParams = {} 
        
        # # Make thins a bit simpler by create 'type' lists
        # paramTypeBool = []
        # paramTypeInt = []
        # paramTypeDouble = []
        # paramTypeString = []
        # paramTypeColor = []
        
       
        # defaultConfigs["TabIndex"] = 0; paramTypeInt.append("TabIndex")
        # defaultConfigs["RecentFile"] = QString(""); paramTypeString.append("RecentFile")
        # defaultConfigs["RecentSimulations"] = []
       
       # # Output tab
        # defaultConfigs["ScreenUpdateFrequency"] = 10; paramTypeInt.append("ScreenUpdateFrequency")
        # defaultConfigs["ImageOutputOn"] = False; paramTypeBool.append("ImageOutputOn")
        # defaultConfigs["SaveImageFrequency"] = 100; paramTypeInt.append("SaveImageFrequency")
        # defaultConfigs["Screenshot_X"] = 600; paramTypeInt.append("Screenshot_X")
        # defaultConfigs["Screenshot_Y"] = 600; paramTypeInt.append("Screenshot_Y")        
        # defaultConfigs["LatticeOutputOn"] = False; paramTypeBool.append("LatticeOutputOn")
        # defaultConfigs["SaveLatticeFrequency"] = 100; paramTypeInt.append("SaveLatticeFrequency")
        # defaultConfigs["GraphicsWinWidth"] = 400; paramTypeInt.append("GraphicsWinWidth")
        # defaultConfigs["GraphicsWinHeight"] = 400; paramTypeInt.append("GraphicsWinHeight")
        # defaultConfigs["UseInternalConsole"] = False; paramTypeBool.append("UseInternalConsole")
        # defaultConfigs["ClosePlayerAfterSimulationDone"] = False; paramTypeBool.append("ClosePlayerAfterSimulationDone")
        
        # # defaultConfigs["ProjectLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DProjects')); paramTypeString.append("ProjectLocation")
        # defaultConfigs["ProjectLocation"] = QString(os.path.join(environ['PREFIX_CC3D'],'Demos')); paramTypeString.append("ProjectLocation")
        
        # defaultConfigs["OutputLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DWorkspace')); paramTypeString.append("OutputLocation")
        # defaultConfigs["OutputToProjectOn"] = False; paramTypeBool.append("OutputToProjectOn")
        # prefsFile = os.path.join(os.path.join(os.path.join(os.path.expanduser('~'),'.config'),ORGANIZATION),APPLICATION+'.ini')
        # prefsFile = APPLICATION
        # defaultConfigs["PreferencesFile"] = QString(prefsFile); paramTypeString.append("PreferencesFile")
        
        # defaultConfigs["NumberOfRecentSimulations"] = 8; paramTypeInt.append("NumberOfRecentSimulations")
        
        
        # # Cells/Colors tab  (used to be: Cell Type tab)
        # defaultConfigs["TypeColorMap"] = { 0:QColor(Qt.black), 1:QColor(Qt.green), 2:QColor(Qt.blue),
            # 3: QColor(Qt.red),
            # 4: QColor(Qt.darkYellow),
            # 5: QColor(Qt.lightGray),
            # 6: QColor(Qt.magenta),
            # 7: QColor(Qt.darkBlue),
            # 8: QColor(Qt.cyan),
            # 9: QColor(Qt.darkGreen),
            # 10: QColor(Qt.white)
            # }
        # defaultConfigs["BorderColor"] = QColor(Qt.yellow); paramTypeColor.append("BorderColor")
        # defaultConfigs["ClusterBorderColor"] = QColor(Qt.blue); paramTypeColor.append("ClusterBorderColor")
        # defaultConfigs["ContourColor"] = QColor(Qt.white); paramTypeColor.append("ContourColor")
        # defaultConfigs["WindowColor"] = QColor(Qt.black); paramTypeColor.append("WindowColor")
        # defaultConfigs["WindowColorSameAsMedium"] = True; paramTypeBool.append("WindowColorSameAsMedium")        
        # defaultConfigs["BrushColor"] = QColor(Qt.white); paramTypeColor.append("BrushColor")
        # defaultConfigs["PenColor"] = QColor(Qt.black); paramTypeColor.append("PenColor")
        
        # defaultConfigs["CellGlyphScaleByVolumeOn"] = False; paramTypeBool.append("CellGlyphScaleByVolumeOn")
        # defaultConfigs["CellGlyphScale"] = 1.0; paramTypeDouble.append("CellGlyphScale")
        # defaultConfigs["CellGlyphThetaRes"] = 2; paramTypeInt.append("CellGlyphThetaRes")
        # defaultConfigs["CellGlyphPhiRes"] = 2; paramTypeInt.append("CellGlyphPhiRes")


        # # Field tab (combines what used to be Colormap tab and Vectors tab)
        
        # defaultConfigs["PixelizedScalarField"] = False; paramTypeBool.append("PixelizedScalarField")
        
        # defaultConfigs["FieldIndex"] = 0; paramTypeInt.append("FieldIndex")
        # defaultConfigs["MinRange"] = 0.0; paramTypeDouble.append("MinRange")
        # defaultConfigs["MinRangeFixed"] = False; paramTypeBool.append("MinRangeFixed")
        # defaultConfigs["MaxRange"] = 1.0; paramTypeDouble.append("MaxRange")
        # defaultConfigs["MaxRangeFixed"] = False; paramTypeBool.append("MaxRangeFixed")
        
        # defaultConfigs["NumberOfLegendBoxes"] = 6; paramTypeInt.append("NumberOfLegendBoxes")
        # defaultConfigs["NumberAccuracy"] = 2; paramTypeInt.append("NumberAccuracy")
        # defaultConfigs["LegendEnable"] = True; paramTypeBool.append("LegendEnable")
        
        # defaultConfigs["ScalarIsoValues"] = QString(" "); paramTypeString.append("ScalarIsoValues")
        # defaultConfigs["NumberOfContourLines"] = 0; paramTypeInt.append("NumberOfContourLines")
# #        defaultConfigs["ContoursOn"] = False; paramTypeBool.append("ContoursOn")
        
        
        # # Vectors tab
        # defaultConfigs["ScaleArrowsOn"] = False; paramTypeBool.append("ScaleArrowsOn")
        # defaultConfigs["ArrowColor"] = QColor(Qt.white); paramTypeColor.append("ArrowColor")
        # defaultConfigs["ArrowLength"] = 1.0; paramTypeDouble.append("ArrowLength")
        # defaultConfigs["FixedArrowColorOn"] = False; paramTypeBool.append("FixedArrowColorOn")
        
        # defaultConfigs["OverlayVectorsOn"] = False; paramTypeBool.append("OverlayVectorsOn")
        
        
        # # 3D tab
        # defaultConfigs["Types3DInvisible"] = QString("0"); paramTypeString.append("Types3DInvisible")
        # defaultConfigs["BoundingBoxOn"] = True; paramTypeBool.append("BoundingBoxOn")
        # defaultConfigs["BoundingBoxColor"] = QColor(Qt.white); paramTypeColor.append("BoundingBoxColor")
        
        
        # #------------- prefs from menu items, etc. (NOT in Preferences dialog) -----------
        # # player layout
        # defaultConfigs["PlayerSizes"] = QByteArray()
        # defaultConfigs["MainWindowSize"] = QSize(900, 650)  # --> VTK winsize of (617, 366)
# #        defaultConfigs["MainWindowSize"] = QSize(1083, 884)  # --> VTK winsize of (800, 600); experiment for Rountree/EPA
        # defaultConfigs["MainWindowPosition"] = QPoint(0,0)
        
        # # visual
        # defaultConfigs["Projection"] = 0; paramTypeInt.append("Projection")
        # defaultConfigs["CellsOn"] = True; paramTypeBool.append("CellsOn")
        # defaultConfigs["CellBordersOn"] = True; paramTypeBool.append("CellBordersOn")
        # defaultConfigs["ClusterBordersOn"] = False; paramTypeBool.append("ClusterBordersOn")
        # defaultConfigs["CellGlyphsOn"] = False; paramTypeBool.append("CellGlyphsOn")
        # defaultConfigs["FPPLinksOn"] = False; paramTypeBool.append("FPPLinksOn")
        # defaultConfigs["FPPLinksColorOn"] = False; paramTypeBool.append("FPPLinksColorOn")
        # defaultConfigs["ConcentrationLimitsOn"] = True; paramTypeBool.append("ConcentrationLimitsOn")
        # defaultConfigs["CC3DOutputOn"] = True; paramTypeBool.append("CC3DOutputOn")
        # defaultConfigs["ZoomFactor"] = 1.0; paramTypeDouble.append("ZoomFactor")
        
        # defaultConfigs["FieldParams"] = {}   # NOTE!!  This needs to be last
        
        # # # # mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, APPLICATION) # use IniFormat instead of NativeFormat now
        
        
        
        # # defaultSettings = defaultSettings()
        # # print 'defaultSettings=',defaultSettings
        # # sys.exit()
        # myGlobalSettings,myGlobalSettingsPath = loadGlobalSettings()
         
        
        # myCustomSettings = None # this is an object that stores settings for custom settings i.e. ones which are associated with individual cc3d projects
        # myCustomSettingsPath = ''
        
        # globalOnlySettings = ['RecentSimulations','NumberOfRecentSimulations']
        
        # activeFieldNamesList = []
# #        mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, "cc3d-2")
# #        initSyncSettings()
# #        updatedConfigs = {}
        
        