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

#def dbgMsg(a1,a2=0,a3=0,a4=0):
#        print a1,
#        if a2: print a2,
#        if a3: print a3,
#        if a4: print a4,
#        print
        
class Configuration():

#    LATTICE_TYPES = {"Square":1,"Hexagonal":2}

#    def __init__(self):

#        LATTICE_TYPES = {"Square":1,"Hexagonal":2}
        #default settings
        defaultConfigs={}
        
        simFieldsParams = {} 
        
        # Make thins a bit simpler by create 'type' lists
        paramTypeBool = []
        paramTypeInt = []
        paramTypeDouble = []
        paramTypeString = []
        paramTypeColor = []
        
#        dbgMsg( MODULENAME, ' >>>>>>>>>>>>>>   __init__  <<<<<<<<<<<<<<<<<<')
#        print MODULENAME, ' >>>>>>>>>>>>>>   __init__  <<<<<<<<<<<<<<<<<<'
       
        defaultConfigs["TabIndex"] = 0; paramTypeInt.append("TabIndex")
        defaultConfigs["RecentFile"] = QString(""); paramTypeString.append("RecentFile")
        defaultConfigs["RecentSimulations"] = []
       
       # Output tab
        defaultConfigs["ScreenUpdateFrequency"] = 10; paramTypeInt.append("ScreenUpdateFrequency")
        defaultConfigs["ImageOutputOn"] = False; paramTypeBool.append("ImageOutputOn")
        defaultConfigs["SaveImageFrequency"] = 100; paramTypeInt.append("SaveImageFrequency")
        defaultConfigs["LatticeOutputOn"] = False; paramTypeBool.append("LatticeOutputOn")
        defaultConfigs["SaveLatticeFrequency"] = 100; paramTypeInt.append("SaveLatticeFrequency")
        defaultConfigs["UseInternalConsole"] = False; paramTypeBool.append("UseInternalConsole")
        defaultConfigs["ClosePlayerAfterSimulationDone"] = False; paramTypeBool.append("ClosePlayerAfterSimulationDone")
        # defaultConfigs["ProjectLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DProjects')); paramTypeString.append("ProjectLocation")
        defaultConfigs["ProjectLocation"] = QString(os.path.join(environ['PREFIX_CC3D'],'Demos')); paramTypeString.append("ProjectLocation")
        
        defaultConfigs["OutputLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DWorkspace')); paramTypeString.append("OutputLocation")
        defaultConfigs["OutputToProjectOn"] = False; paramTypeBool.append("OutputToProjectOn")
        prefsFile = os.path.join(os.path.join(os.path.join(os.path.expanduser('~'),'.config'),ORGANIZATION),APPLICATION+'.ini')
        prefsFile = APPLICATION
        defaultConfigs["PreferencesFile"] = QString(prefsFile); paramTypeString.append("PreferencesFile")
        
        
        # Cell Type tab
        defaultConfigs["TypeColorMap"] = { 0:QColor(Qt.black), 1:QColor(Qt.green), 2:QColor(Qt.blue),
            3: QColor(Qt.red),
            4: QColor(Qt.darkYellow),
            5: QColor(Qt.lightGray),
            6: QColor(Qt.magenta),
            7: QColor(Qt.darkBlue),
            8: QColor(Qt.cyan),
            9: QColor(Qt.darkGreen),
            10: QColor(Qt.white)
            }
        defaultConfigs["BorderColor"] = QColor(Qt.yellow); paramTypeColor.append("BorderColor")
        defaultConfigs["ClusterBorderColor"] = QColor(Qt.blue); paramTypeColor.append("ClusterBorderColor")
        defaultConfigs["ContourColor"] = QColor(Qt.white); paramTypeColor.append("ContourColor")
        defaultConfigs["WindowColor"] = QColor(Qt.black); paramTypeColor.append("WindowColor")
        defaultConfigs["BrushColor"] = QColor(Qt.white); paramTypeColor.append("BrushColor")
        defaultConfigs["PenColor"] = QColor(Qt.black); paramTypeColor.append("PenColor")
        
        defaultConfigs["CellGlyphScale"] = 1.0; paramTypeDouble.append("CellGlyphScale")
        defaultConfigs["CellGlyphThetaRes"] = 2; paramTypeInt.append("CellGlyphThetaRes")
        defaultConfigs["CellGlyphPhiRes"] = 2; paramTypeInt.append("CellGlyphPhiRes")


        # Field tab (combines what used to be Colormap tab and Vectors tab)
        defaultConfigs["FieldIndex"] = 0; paramTypeInt.append("FieldIndex")
        defaultConfigs["MinRange"] = 0.0; paramTypeDouble.append("MinRange")
        defaultConfigs["MinRangeFixed"] = False; paramTypeBool.append("MinRangeFixed")
        defaultConfigs["MaxRange"] = 1.0; paramTypeDouble.append("MaxRange")
        defaultConfigs["MaxRangeFixed"] = False; paramTypeBool.append("MaxRangeFixed")
        
        defaultConfigs["NumberOfLegendBoxes"] = 6; paramTypeInt.append("NumberOfLegendBoxes")
        defaultConfigs["NumberAccuracy"] = 2; paramTypeInt.append("NumberAccuracy")
        defaultConfigs["LegendEnable"] = True; paramTypeBool.append("LegendEnable")
        
        defaultConfigs["ScalarIsoValues"] = QString(" "); paramTypeString.append("ScalarIsoValues")
        defaultConfigs["NumberOfContourLines"] = 5; paramTypeInt.append("NumberOfContourLines")
#        defaultConfigs["ContoursOn"] = False; paramTypeBool.append("ContoursOn")
        
        
        # Vectors tab
#        defaultConfigs["MinMagnitude"] = 0.0; paramTypeDouble.append("MinMagnitude")
#        defaultConfigs["MinMagnitudeFixed"] = False; paramTypeBool.append("MinMagnitudeFixed")
#        defaultConfigs["MaxMagnitude"] = 1.0; paramTypeDouble.append("MaxMagnitude")
#        defaultConfigs["MaxMagnitudeFixed"] = False; paramTypeBool.append("MaxMagnitudeFixed")
#        
#        defaultConfigs["NumberOfLegendBoxesVector"] = 6; paramTypeInt.append("NumberOfLegendBoxesVector")
#        defaultConfigs["NumberAccuracyVector"] = 2; paramTypeInt.append("NumberAccuracyVector")
#        defaultConfigs["LegendEnableVector"] = True; paramTypeBool.append("LegendEnableVector")
        
        defaultConfigs["ScaleArrowsOn"] = False; paramTypeBool.append("ScaleArrowsOn")
        defaultConfigs["ArrowColor"] = QColor(Qt.white); paramTypeColor.append("ArrowColor")
        defaultConfigs["ArrowLength"] = 1.0; paramTypeDouble.append("ArrowLength")
        defaultConfigs["FixedArrowColorOn"] = False; paramTypeBool.append("FixedArrowColorOn")
        
        defaultConfigs["OverlayVectorsOn"] = False; paramTypeBool.append("OverlayVectorsOn")
        
        
        # 3D tab
        defaultConfigs["Types3DInvisible"] = QString("0"); paramTypeString.append("Types3DInvisible")
        defaultConfigs["BoundingBoxOn"] = True; paramTypeBool.append("BoundingBoxOn")
        
        
        #------------- prefs from menu items, etc. (NOT in Preferences dialog) -----------
        # player layout
        defaultConfigs["PlayerSizes"] = QByteArray()
        defaultConfigs["MainWindowSize"] = QSize(900, 650)  # --> VTK winsize of (617, 366)
#        defaultConfigs["MainWindowSize"] = QSize(1083, 884)  # --> VTK winsize of (800, 600); experiment for Rountree/EPA
        defaultConfigs["MainWindowPosition"] = QPoint(0,0)
        
        # visual
        defaultConfigs["Projection"] = 0; paramTypeInt.append("Projection")
        defaultConfigs["CellsOn"] = True; paramTypeBool.append("CellsOn")
        defaultConfigs["CellBordersOn"] = True; paramTypeBool.append("CellBordersOn")
        defaultConfigs["ClusterBordersOn"] = False; paramTypeBool.append("ClusterBordersOn")
        defaultConfigs["CellGlyphsOn"] = False; paramTypeBool.append("CellGlyphsOn")
        defaultConfigs["FPPLinksOn"] = False; paramTypeBool.append("FPPLinksOn")
        defaultConfigs["FPPLinksColorOn"] = False; paramTypeBool.append("FPPLinksColorOn")
        defaultConfigs["ConcentrationLimitsOn"] = True; paramTypeBool.append("ConcentrationLimitsOn")
        defaultConfigs["CC3DOutputOn"] = True; paramTypeBool.append("CC3DOutputOn")
        defaultConfigs["ZoomFactor"] = 1.0; paramTypeDouble.append("ZoomFactor")
        
#        defaultConfigs["CurrentFieldName"] = QString("0"); paramTypeString.append("CurrentFieldName")
        defaultConfigs["FieldParams"] = {}   # NOTE!!  This needs to be last
        
#        foo=1/0
#        modifiedKeyboardShortcuts={} # dictionary actionName->shortcut for modified keyboard shortcuts - only reassigned shortcuts are stored
#        print MODULENAME,'-------------   doing QSettings -------------------\n'
        mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, APPLICATION) # use IniFormat instead of NativeFormat now
#        print MODULENAME,'    type(mySettings)= ',type(mySettings)   # <class 'PyQt4.QtCore.QSettings'>
        
#        mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, "cc3d-2")
#        initSyncSettings()
#        updatedConfigs = {}
        
        
#def getLatticeTypes(:
#        return LATTICE_TYPES.keys()
    
#def getLatticeTypeVal(_key):
#        dbgMsg MODULENAME, '  ------- getLatticeTypeVal'
##        if _key in ["Square","Hexagonal"]:
#        return LATTICE_TYPES[_key]
        
def setPrefsFile(fname):
    print
    print MODULENAME,'------------  setPrefsFile:  fname=',fname,'\n'
    Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, fname)

def getVersion():
    import Version
    return Version.getVersionAsString()
    
def getSimFieldsParams():
#    return simFieldsParams   # global name 'simFieldsParams' is not defined
    fieldParams = Configuration.simFieldsParams
#    print
#    print MODULENAME,'   getSimFieldsParams:  Configuration.simFieldsParams=',fieldParams
    if len(Configuration.simFieldsParams) == 0:
#        print MODULENAME,'   getSimFieldsParams:  EMPTY Configuration.simFieldsParams !!!, try to getSetting(FieldParams)'
        fieldParams = getSetting('FieldParams')
        Configuration.simFieldsParams = fieldParams 
#        print MODULENAME,'   getSimFieldsParams:  getSetting(FieldParams)=',fieldParams
#    print '-------------------   end of getSimFieldsParams  ---------------'
#    print
#    return Configuration.simFieldsParams
    return fieldParams

#def setSimFieldsParams(fieldNames):
def initFieldsParams(fieldNames):   # called from SimpleTabView once we know the fields
    print
#    print MODULENAME,'============================  start of initFieldsParams  ======================'
#    print MODULENAME,'   initFieldsParams:  fieldNames=',fieldNames
    fieldParams = getSetting('FieldParams')
#    print MODULENAME,'   initFieldsParams:  FieldParams (from getSetting)',fieldParams
#    print MODULENAME,'   initFieldsParams:  type(FieldParams)',type(fieldParams)   # dict
    for field in fieldNames:
#        print MODULENAME,'  initFieldsParams:   consider field=',field
        if field not in fieldParams.keys() and field != 'Cell_Field':
            paramsDict = {}
#            print MODULENAME,'  initFieldsParams:   adding field=',field
            paramsDict["MinRange"] = getSetting("MinRange")
            paramsDict["MinRangeFixed"] = getSetting("MinRangeFixed")
            paramsDict["MaxRange"] = getSetting("MaxRange")
            paramsDict["MaxRangeFixed"] = getSetting("MaxRangeFixed")
            
            paramsDict["NumberOfLegendBoxes"] = getSetting("NumberOfLegendBoxes")
            paramsDict["NumberAccuracy"] = getSetting("NumberAccuracy")
            paramsDict["LegendEnable"] = getSetting("LegendEnable")
        
            paramsDict["NumberOfContourLines"] = getSetting("NumberOfContourLines")
#            paramsDict["ContoursOn"] = getSetting("ContoursOn")
        
            paramsDict["ScaleArrowsOn"] = getSetting("ScaleArrowsOn")
            color = getSetting("ArrowColor")
#            print MODULENAME,'   initFieldsParams:  type(color),color=',type(color),color
            paramsDict["ArrowColor"] = color
#            paramsDict["ArrowColor"] = (color.red(),color.green(),color.blue())
            paramsDict["ArrowLength"] = getSetting("ArrowLength")
            paramsDict["FixedArrowColorOn"] = getSetting("FixedArrowColorOn")
            paramsDict["OverlayVectorsOn"] = getSetting("OverlayVectorsOn")
        
#            fieldParams[field] = 'stuff'
            fieldParams[field] = paramsDict
        
#            print MODULENAME,'  initFieldsParams:    paramsDict=',paramsDict
    
#    print MODULENAME,'   initFieldsParams:  new fieldParams',fieldParams
#    print MODULENAME,'   type(fieldParams)',type(fieldParams)
    Configuration.simFieldsParams = fieldParams
#    print
#    print MODULENAME,'   initFieldsParams:  Configuration.simFieldsParams=',Configuration.simFieldsParams
#    print '-------------------'
#    print
    setSetting('FieldParams',fieldParams )
    # if not defined in Prefs yet, init params for all fields
#    print MODULENAME,'============================  end of initFieldsParams  ======================'
#    print
    
def updateFieldsParams(fieldName,fieldDict):
#    print
#    print MODULENAME,' updateFieldsParams():  field,dict=',fieldName,fieldDict
#    print MODULENAME,' updateFieldsParams():  Configuration.simFieldsParams=',Configuration.simFieldsParams
    
    fieldParamsDict = getSetting("FieldParams")
#    print
#    print MODULENAME,' updateFieldsParams():  from getSetting(FieldParams) =',fieldParamsDict
#    print
#    print MODULENAME,' =========>  updateFieldsParams():  type(fieldParamsDict.keys()[0])=',type(fieldParamsDict.keys()[0])
#    if fieldName in Configuration.simFieldsParams.keys():
#        print '-----------> field is already in simFieldsParams.keys()'
#    if fieldName not in Configuration.simFieldsParams.keys():

    if not isinstance(fieldName,str):
#        print MODULENAME,' updateFieldsParams():  fieldName is not a str, converting to str'
        fieldName = str(fieldName)

#    if isinstance(fieldName,str):
#        print MODULENAME,' updateFieldsParams():  fieldName is a str, converting to QString'
#        fieldName = QString(fieldName)
        
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not
#    print
#    print MODULENAME,' updateFieldsParams():  post Configuration.simFieldsParams=',Configuration.simFieldsParams
    setSetting('FieldParams',Configuration.simFieldsParams)
    
    
def getSetting(_key, fieldName=None):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
#        print '\n------------------'
#        print MODULENAME, '  ------- getSetting, _key, fieldName = ', _key,fieldName
        if fieldName:
            if fieldName == 'Cell_Field':  # if there are no fields defined, but just the Cell_Field, return default Pref (hard-coded -> BAD)
                return getSetting(_key)
            
#            print MODULENAME, ' ------------>  getSetting():  fieldName,_key=',fieldName,_key
            fieldsDict = getSimFieldsParams()
#            print MODULENAME, ' ------------>  getSetting():  fieldsDict=',fieldsDict
            paramsDict = fieldsDict[fieldName]
#            print MODULENAME, ' ------------>  getSetting():  paramsDict=',paramsDict
            if _key == 'ArrowColor':  
#                dictVals[str(key2)] = str(dict2[key2].toString())
                val = paramsDict[_key]
#                print MODULENAME, ' ------------>  getSetting(): w/ fieldName, ArrowColor: type(val),val=',type(val),val
                r, g, b = (1,0,0)
                if isinstance(val,str) and val[0]=='#':  #   hex value, e.g.  '#ff0000   (i.e. #rrggbb)
#                    print MODULENAME, ' ------------>  getSetting(): w/ fieldName, ArrowColor: got hex string'
                    r, g, b = val[1:3], val[3:5], val[5:]
                    r, g, b = [int(n, 16)/255. for n in (r, g, b)]   # return normalized in [0,1] for VTK
                else:
#                    print MODULENAME, ' ------------>  getSetting(): w/ fieldName, ArrowColor: got QColor?'
                    r= val.red()/255.
                    g= val.green()/255.
                    b= val.blue()/255.
#                    raise
#                    val = QColor(paramsDict[_key].toString())
#                print MODULENAME, ' ------------>  getSetting(): w/ fieldName, ArrowColor: r,g,b=',r,g,b
                return (r,g,b) 
            else:
                if _key not in paramsDict.keys():
                    print MODULENAME, ' ------------>  WARNING:  getSetting(): _key not in paramsDict',_key,paramsDict
                    return 0
                val = paramsDict[_key]
#                print MODULENAME, ' ------------>  getSetting(): w/ fieldName, type(val),val=',fieldName,type(val),val
                return val 
        elif _key in Configuration.paramTypeBool:
#            if _key == 'LegendEnable': v=1/0
            val = Configuration.mySettings.value(_key)
#            dbgMsg MODULENAME, ' _key,val, val.isValid=',_key,val,val.isValid()
            if val.isValid():
                return val.toBool()
            else:
                return Configuration.defaultConfigs[_key]
        
        elif _key in Configuration.paramTypeString:
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                return val.toString()
            else:
                return Configuration.defaultConfigs[_key]
        
        elif _key in Configuration.paramTypeInt:   # ["ScreenUpdateFrequency","SaveImageFrequency"]:
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                return val.toInt()[0] # toInt returns tuple: first = integer; second = flag
            else:
                return Configuration.defaultConfigs[_key]
            
        elif _key in Configuration.paramTypeDouble:
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                return val.toDouble()[0]
            else:
                return Configuration.defaultConfigs[_key]
            
        elif _key in Configuration.paramTypeColor:
            val = Configuration.mySettings.value(_key)
#            print MODULENAME, '  ------- getSetting, _key,val = ', _key,val
            if val.isValid():
                color = QColor(val.toString())
#                color = QColor(val)
#                print MODULENAME, '  ------- getSetting, (valid) color= ', color
#                print MODULENAME, '  ------- getSetting, (valid) color rgb= ', color.red(),color.green(),color.blue()
                return color
            else:
                color = Configuration.defaultConfigs[_key]
#                print MODULENAME, '  ------- getSetting, (invalid) default color= ', color
                return color
        
        elif _key in ["RecentSimulations"]:
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
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                fieldDict = val.toMap()
#                print
#                print MODULENAME,'getSetting():  valid FieldParams, fieldDict=',fieldDict
#                print MODULENAME,'getSetting():  valid FieldParams, fieldDict.keys()=',fieldDict.keys()
#                print MODULENAME,'getSetting():  valid FieldParams, isinstance(fieldDict.keys()[0],str)=',isinstance(fieldDict.keys()[0],str)
                fieldDictNew = {}
                knt = 0
                for key in fieldDict.keys():
#                    print MODULENAME,'getSetting():  ------------  key=',key
                    fieldDictNew[str(key)] = {}
                    val = fieldDict.values()[knt]
#                    print MODULENAME,'getSetting():  valid FieldParams, fieldDict knt,val=',knt,val
#                    print MODULENAME,'getSetting():  type(val)=',type(val)
                    dict2 = val.toMap()
#                    print MODULENAME,'getSetting():  val.toMap()=',dict2
                    dictVals = {}
                    for key2 in dict2.keys():
#                        print MODULENAME,'-------  str(key2)=',str(key2)
#-------  str(key2)= ArrowColor
#-------  str(key2)= NumberOfContourLines
#-------  str(key2)= LegendEnable
#-------  str(key2)= ArrowLength
#-------  str(key2)= NumberAccuracy
#-------  str(key2)= ContoursOn
#-------  str(key2)= NumberOfLegendBoxes
#-------  str(key2)= MaxRangeFixed
#-------  str(key2)= MaxRange
#-------  str(key2)= MinRangeFixed
#-------  str(key2)= OverlayVectorsOn
#-------  str(key2)= ScaleArrowsOn
#-------  str(key2)= FixedArrowColorOn
#-------  str(key2)= MinRange
                        if str(key2)[-2:] == 'On':  dictVals[str(key2)] = dict2[key2].toBool()
                        elif str(key2)[-5:] == 'Fixed':  dictVals[str(key2)] = dict2[key2].toBool()
                        elif str(key2) == 'LegendEnable':  dictVals[str(key2)] = dict2[key2].toBool()
                        elif str(key2)[:6] == 'Number':  dictVals[str(key2)] = dict2[key2].toInt()[0]  # e.g. toInt() -> (3,True)
                        elif str(key2)[-6:] == 'Length':  dictVals[str(key2)] = dict2[key2].toInt()[0]
                        elif str(key2)[-5:] == 'Range':  dictVals[str(key2)] = dict2[key2].toFloat()[0]
                        elif str(key2)[-4:] == 'List':  dictVals[str(key2)] = dict2[key2].toString()[0]
                        elif str(key2) == 'ArrowColor':  
#                            dictVals[str(key2)] = dict2[key2].toString()
                            dictVals[str(key2)] = str(dict2[key2].toString())
                            mycolor = QColor(dict2[key2].toString())
#                            print '     mycolor RGB=',mycolor.red(),mycolor.green(),mycolor.blue()
                        else:  dictVals[str(key2)] = dict2[key2]
#                    print MODULENAME,'getSetting():  dictVals=',dictVals
                    fieldDictNew[str(key)] = dictVals
                    knt += 1
#                print
#                print MODULENAME,'getSetting():  FieldParams: fieldDictNew=',fieldDictNew
#                print
#                recentSimulations=[]
#                for i in range(recentSimulationsList.count()):
#                    recentSimulations.append(str(recentSimulationsList[i]))
#                return fieldDict
                return fieldDictNew
            else:
                fieldDict = Configuration.defaultConfigs[_key]
#                print
#                print MODULENAME,'getSetting():  invalid FieldParams, fieldDict=',fieldDict
                # rwh: need to call initFieldsParams
                return fieldDict
        elif _key in ["MainWindowSize","InitialSize"]: # QSize values
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                return val.toSize() 
            else:
                return Configuration.defaultConfigs[_key]                             

        elif _key in ["MainWindowPosition","InitialPosition"]: # QPoint values
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                pval = val.toPoint()
#                print MODULENAME, 'getSetting(), val isValid, type(pval), pval=',type(pval), pval
                return val.toPoint() 
            else:
                return Configuration.defaultConfigs[_key]
            
        elif _key in ["PlayerSizes"]:
            val = Configuration.mySettings.value(_key)
            if val.isValid():
                return val.toByteArray() 
            else:
                return Configuration.defaultConfigs[_key]
            
        elif _key in ["TypeColorMap"]:
#            print MODULENAME, '  ------- getSetting, _key = ', _key
            colorMapStr = Configuration.mySettings.value(_key)
    
            if colorMapStr.isValid():
                colorList = colorMapStr.toStringList()
                if colorList.count() == 0:
                    # colorList = prefClass.colorsDefaults[key]
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
#                print MODULENAME, '  ------- return colorDict '
                return colorDict
            else:
#                print MODULENAME, '  ------- return default typecolormap!! '
                return Configuration.defaultConfigs["TypeColorMap"]
            
                
#        elif _key in ["ListOfOpenFiles","FRFindHistory","FRReplaceHistory","FRFiltersHistory","FRDirectoryHistory","KeyboardShortcuts"]: # QStringList values
#            val = Configuration.mySettings.value(_key)
#            if val.isValid():
#                return val.toStringList() 
#            else:
#                return defaultConfigs[_key]

        else:
            print MODULENAME,' getSetting(), bogus key =',_key
            raise # exception

def addNewSimulation(recentSimulationsList,value):
#    print MODULENAME,'  addNewSimulation:  value=',value
#    print MODULENAME,'  addNewSimulation:  str(value)=',str(value)
#    dbgMsg( MODULENAME,'  addNewSimulation:  str(value)=',str(value) )
#    print MODULENAME,'  addNewSimulation:  value.toString()=',value.toString()
    if str(value)=="":
        return False
    elementExists=False
    idxFound = -1
    for idx in range(recentSimulationsList.count()):
#        print "recentSimulationsList[idx]=",recentSimulationsList[idx]
#        print "value=",value
        if str(recentSimulationsList[idx])==value:
            elementExists=True
            idxFound=idx
            break
    if not elementExists:    
#        print '  --- addNewSimulation (dne):  str(value)=',value
        recentSimulationsList.prepend(str(value))
        return True
    else:
        # moving existing item to the beginning of the list
        fileNameTmp = recentSimulationsList[idxFound]
#        print '  --- addNewSimulation:  fileNameTmp=',fileNameTmp
        recentSimulationsList.removeAt(idxFound)
        recentSimulationsList.prepend(fileNameTmp)
        return False
                
def setSetting(_key,_value):  # rf. ConfigurationDialog.py, updatePreferences()
#        print MODULENAME, '  --- setSetting: _key,_value=',_key,_value
        if _key in Configuration.paramTypeBool:
            Configuration.mySettings.setValue(_key,QVariant(_value))
        elif _key in Configuration.paramTypeInt:
            Configuration.mySettings.setValue(_key,_value)
        elif _key in Configuration.paramTypeDouble:
#            print MODULENAME,' setSetting: _key,_value=',_key,_value
            Configuration.mySettings.setValue(_key,QVariant(_value))
        elif _key in Configuration.paramTypeString:
#            if _key == 'PreferencesFile':
#                print MODULENAME,' setSetting: ',_key,_value
#                print MODULENAME,' type(_key),type(_value)=',type(_key),type(_value)
#                print MODULENAME,' str(_value)=',str(_value)
#                print MODULENAME,' dir(_value)=',dir(_value)
#                try:
#                    prefsFile = _value.toString()
##                    print ' _value.toString() =',_value.toString()
##                    print ' type(_value.toString()) =',type(_value.toString())
#                    Configuration.setPrefsFile(prefsFile)
#                except:
#                    print '  -----> ERROR:  cannot do _value.toString()'
#                try:
#                    print ' _value.toPyObject() =',_value.toPyObject()
#                    print ' type(_value.toPyObject()) =',type(_value.toPyObject())
##                    Configuration.setPrefsFile(prefsFile)
#                except:
#                    print '  -----> ERROR:  cannot do _value.toPyObject()'
#            print MODULENAME,' setSetting (toString): ',_key,_value.toString()
            Configuration.mySettings.setValue(_key,_value)
        elif _key in Configuration.paramTypeColor:
#            print MODULENAME,' -----setSetting(): paramTypeColor: ',_key,_value
#            if _key == 'BorderColor':
#              print MODULENAME,' -----setSetting(): BorderColor dir(_value): ',dir(_value)
            Configuration.mySettings.setValue(_key,_value)
#            Configuration.mySettings.setValue(_key,QVariant(_value))
#            Configuration.mySettings.setValue(_key,QColor(_value))
        
        elif _key in ["RecentSimulations"]:
#            print MODULENAME,' setSetting:RecentSimulations:  _key,_value= ',_key,_value
#            return
            recentSimulationsVariant = Configuration.mySettings.value("RecentSimulations")
            if recentSimulationsVariant.isValid():
                recentSimulationsList = recentSimulationsVariant.toStringList()
#                print "setSetting:  recentSimulationsList.count()=",recentSimulationsList.count()
#                print "setSetting: maxNumberOfRecentFiles=",maxNumberOfRecentFiles
                if recentSimulationsList.count() >= maxNumberOfRecentFiles:    
                    addingSuccessful = addNewSimulation(recentSimulationsList,_value)
                    if addingSuccessful:
                        recentSimulationsList.removeAt(recentSimulationsList.count()-1)
                else:
#                    print 'setSetting:    addNewSim... _value = ',_value
#                    print '    addNewSim... _value.toString() = ',_value.toString() # AttributeError: 'str' object has no attribute 'toString'
                    addingSuccessful = addNewSimulation(recentSimulationsList,_value)
        
                if addingSuccessful:
#                    print "setSetting: doing setValue for recentSim-List="
                    Configuration.mySettings.setValue("RecentSimulations", QVariant(recentSimulationsList))
                    
#                for idx in range(recentSimulationsList.count()):
#                    print "READ EXISTING FILE  =",idx," file name=", recentSimulationsList[idx]
            else:
#                print "       recentSimulationsVariant is NOT valid:  _key,_value=",_key,_value
                recentSimulationsList = QStringList()
#                recentSimulationsList.prepend(QString(_value))
                Configuration.mySettings.setValue("RecentSimulations", QVariant(recentSimulationsList))
            
        # string
        elif _key in ["BaseFontName","BaseFontSize"]:
            Configuration.mySettings.setValue(_key,QVariant(_value))
        
        # QSize, QPoint,QStringList , QString
        elif _key in ["InitialSize","InitialPosition","KeyboardShortcuts"]:
            Configuration.mySettings.setValue(_key,QVariant(_value))
            
        elif _key in ["PlayerSizes","MainWindowSize","MainWindowPosition"]:
            Configuration.mySettings.setValue(_key, QVariant(_value))
            
        elif _key in ["FieldParams"]:
#            print MODULENAME,' setSetting FieldParams:  _key,_value= ',_key,_value
#            print MODULENAME,' setSetting FieldParams:  _key,type(_value)= ',_key,type(_value)
            if isinstance(_value,dict):
#                print MODULENAME,' setSetting FieldParams (dict):  _value.keys()= ',_value.keys()
#                print MODULENAME,' setSetting FieldParams (dict):  _value.values()= ', _value.values()
                Configuration.mySettings.setValue(_key, QVariant(_value))
            else:  # this block gets executed, it seems
                valDict = _value.toMap()
#                print MODULENAME,' setSetting FieldParams:  type(valDict)= ',type(valDict)
#                print MODULENAME,' setSetting FieldParams:  valDict.keys()= ',valDict.keys()
#                print MODULENAME,' setSetting FieldParams:  valDict.values()= ', valDict.values()
#                val1 = valDict.values()[0]
#                print MODULENAME,' setSetting FieldParams:  val1.toMap()= ', val1.toMap()
#                for key2 in val1.toMap().keys():
#                        print '-------  str(key2)=',str(key2)
#-------  str(key2)= ArrowColor
#-------  str(key2)= NumberOfContourLines
#-------  str(key2)= LegendEnable
#-------  str(key2)= ArrowLength
#-------  str(key2)= NumberAccuracy
#-------  str(key2)= ContoursOn
#-------  str(key2)= NumberOfLegendBoxes
#-------  str(key2)= MaxRangeFixed
#-------  str(key2)= MaxRange
#-------  str(key2)= MinRangeFixed
#-------  str(key2)= OverlayVectorsOn
#-------  str(key2)= ScaleArrowsOn
#-------  str(key2)= FixedArrowColorOn
#-------  str(key2)= MinRange
#                    if str(key2) == 'MinRange':      print '     setSetting():   MinRange value = ',val1.toMap()[key2].toFloat()
#                    elif str(key2) == 'ArrowColor':  
#                        print '     setSetting():   ArrowColor value = ',val1.toMap()[key2]
#                        print '     setSetting():   ArrowColor value .toString = ',val1.toMap()[key2].toString()
                        
                Configuration.mySettings.setValue(_key, QVariant(valDict))
            
#            paramList = QStringList()
#            if type(_value) == dict:
#                print MODULENAME,' setSetting:   yes, _value is a dict'
#                print MODULENAME,' setSetting: _key=TypeColorMap: len(_value) =',len(_value)
#                for i in range(len(_value)):
#                    keys = _value.keys()
#                    paramList.append(str(keys[i]))
#                    paramList.append(str(_value[keys[i]].name()))
#                    print i,str(keys[i]),str(_value[keys[i]].name())
#                    
#                Configuration.mySettings.setValue(_key, QVariant(paramList))
                
            # convert QVariant to Python dict
#            retval = Configuration.mySettings.value(_key)
#            fieldDict = retval.toMap()
#            print MODULENAME,' setSetting: fieldDict (before) = ',fieldDict
#            # need to get currently selected fieldname in graphics frame
#            currentField = Configuration.mySettings.value("CurrentFieldName")
#            print MODULENAME,' setSetting: currentField = ',currentField 
#            currentField = currentField.toString()
#            print MODULENAME,' setSetting: currentField.toString() = ',currentField
#            if currentField in fieldDict.keys():
#                print MODULENAME,' setSetting: FieldParams:  yes, ',_value,' is in ',fieldDict,' --> need to update!'
#                print MODULENAME,'    this is previous field params:  ',fieldDict[currentField].toMap()
#            else:
#                print MODULENAME,' setSetting: FieldParams:  no, ',_value,' is not in ',fieldDict
#            fieldParamsDict = {}
#            fieldParamsDict['scalarMin'] = Configuration.mySettings.value("MinRange").toFloat()[0]
#            fieldParamsDict['scalarMinFixed'] = Configuration.mySettings.value("MinRangeFixed").toBool()
#            fieldParamsDict['scalarMax'] = Configuration.mySettings.value("MaxRange").toFloat()[0]
#            fieldParamsDict['scalarMaxFixed'] = Configuration.mySettings.value("MaxRangeFixed").toBool()
#            fieldParamsDict['numScalarLabels'] = Configuration.mySettings.value("NumberOfLegendBoxes").toInt()[0]
#            fieldParamsDict['numScalarDigits'] = Configuration.mySettings.value("NumberAccuracy").toInt()[0]
#            fieldParamsDict['scalarMapFlag'] = Configuration.mySettings.value("LegendEnable").toBool()
#            fieldParamsDict['numContourLines'] = Configuration.mySettings.value("NumberOfContourLines").toInt()[0]
#            fieldParamsDict['contoursFlag'] = Configuration.mySettings.value("ContoursOn").toBool()
#        
#        
##            fieldDict[currentField] = {'smin':-42.0,'smax':42.0}
#            fieldDict[currentField] = fieldParamsDict    # fieldDict is a dict of dicts
#                
#            print MODULENAME,' setSetting: fieldDict (after) = ',fieldDict
#            Configuration.mySettings.setValue(_key, fieldDict)
                
            
        elif _key in ["TypeColorMap"]:
#            print MODULENAME,' setSetting:TypeColorMap= '
#            print 'type(_value) = ',type(_value)   # <type 'dict'> or <class 'PyQt4.QtCore.QVariant'>
#            print '_value =',_value
            penColorList = QStringList()
#            print '---  Config-/__init__.py: penColorList =',penColorList
            
            if type(_value) == dict:
#                print 'yes, _value is a dict'
#                print '---  Config-/__init__.py: setSetting: _key=TypeColorMap: len(_value) =',len(_value)
                for i in range(len(_value)):
                    keys = _value.keys()
                    penColorList.append(str(keys[i]))
                    penColorList.append(str(_value[keys[i]].name()))
#                    print i,str(keys[i]),str(_value[keys[i]].name())
                    
                Configuration.mySettings.setValue(_key, QVariant(penColorList))
 
            # rwh: I confess I'm confused, but it seems this block is not even needed
#            else:            # QVariant
#            dbgMsg ('     >>>> TODO! setSetting TypeColorMap')
#            for i in range(len(_value)):
#            for i in range(len(QVariant(_value))):
#            keys = Configuration.defaultConfigs["TypeColorMap"].keys()
#            print MODULENAME,' TypeColorMap keys= ',keys
#            for i in range(len(Configuration.defaultConfigs["TypeColorMap"])):
#            for i in range(len(dict(_value))):  #  _value is a dict

#                clist = _value.toStringList()
#                numColors = len(clist)
#                print 'numColors = ',numColors
#                for i in range(len(_value.toStringList())):  #  _value is a QVariant
#                    print i,clist[i]

#                keys = _value.keys()
#                penColorList.append(str(keys[i]))
#                penColorList.append(str(_value[keys[i]].name()))
                
#            Configuration.mySettings.setValue(_key, QVariant(penColorList))

        
        else:
#            dbgMsg("Wrong format of configuration option:",_key,":",_value)
            print MODULENAME,"Wrong format of configuration option:" + str(_key) + ":" + str(_value)
            
#    def setKeyboardShortcut(_actionName,_keyboardshortcut):
#        modifiedKeyboardShortcuts[_actionName]=_keyboardshortcut    
#        
#    def keyboardShortcuts(:
#        return modifiedKeyboardShortcuts
#    
#    def prepareKeyboardShortcutsForStorage(:
#        modifiedKeyboardShortcutsStringList=QStringList()
#        for actionName in modifiedKeyboardShortcuts.keys():
#            modifiedKeyboardShortcutsStringList.append(actionName)
#            modifiedKeyboardShortcutsStringList.append(self.modifiedKeyboardShortcuts[actionName])
#            
#        setSetting("KeyboardShortcuts",self.modifiedKeyboardShortcutsStringList)    

def getPlayerParams():
#    print MODULENAME,' getPlayerParams() called'
    playerParamsDict = {}
    for key in Configuration.defaultConfigs.keys():
        if key not in ["PlayerSizes"]:
            playerParamsDict[key] = getSetting(key)

#        dbgMsg 'playerParamsDict=',playerParamsDict
    return playerParamsDict
    
def syncPreferences():
#    print
#    print MODULENAME,'----------- syncPreferences -------------'
    for key in Configuration.defaultConfigs.keys():
        val = Configuration.mySettings.value(key)
#        print 'key: type(key,val,Configuration.defaultConfigs[key] = ',key,type(key),type(val),type(Configuration.defaultConfigs[key])
        if val.isValid():  # if setting exists (is valid) in the .plist
            if not key == 'RecentSimulations':
#                if key == 'FieldParams': 
#                    print '      syncPreferences (valid):  calling setSetting on FieldParams: key,val=',key,val
                setSetting(key,val)
        else:
#            if key=="TypeColorMap":
#                setSetting(key,QVariant(Configuration.defaultConfigs[key]))
#            else:
#            if key == 'FieldParams': 
#                print '      syncPreferences (not valid):  calling setSetting on FieldParams: key,val=',key,Configuration.defaultConfigs[key]
            setSetting(key,Configuration.defaultConfigs[key])
            
#    print MODULENAME,'----------- leaving syncPreferences -------------'
#    print
                
        # initialize modifiedKeyboardShortcuts
#        modifiedKeyboardShortcutsStringList=self.setting("KeyboardShortcuts")
#        for i in range(0,modifiedKeyboardShortcutsStringList.count(),2): 
#            modifiedKeyboardShortcuts[str(self.modifiedKeyboardShortcutsStringList[i])]=str(self.modifiedKeyboardShortcutsStringList[i+1])        

                
#    def syncPreferences(prefClass = Configs):
##    Module function to sync the preferences to disk.
##    In addition to syncing, the central configuration store is reinitialized as well.
##    @param prefClass preferences class used as the storage area
#        dbgMsg MODULENAME," syncPreferences(), WILL WRITE SETTINGS"
#    # import time
#    # time.sleep(5)
#    
#        prefClass.mySettings.setValue("General/Configured", QVariant(1))
#        initPreferences()
                
        # initialize modifiedKeyboardShortcuts
#        modifiedKeyboardShortcutsStringList=self.setting("KeyboardShortcuts")
#        for i in range(0,modifiedKeyboardShortcutsStringList.count(),2): 
#            modifiedKeyboardShortcuts[str(self.modifiedKeyboardShortcutsStringList[i])] = 
#                  str(modifiedKeyboardShortcutsStringList[i+1])
        
        
#syncPreferences()   # don't think needed; rf. UI/