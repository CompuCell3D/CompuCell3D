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

        
class Configuration():

        #default settings
        defaultConfigs={}
        
        simFieldsParams = {} 
        
        # Make thins a bit simpler by create 'type' lists
        paramTypeBool = []
        paramTypeInt = []
        paramTypeDouble = []
        paramTypeString = []
        paramTypeColor = []
        
       
        defaultConfigs["TabIndex"] = 0; paramTypeInt.append("TabIndex")
        defaultConfigs["RecentFile"] = QString(""); paramTypeString.append("RecentFile")
        defaultConfigs["RecentSimulations"] = []
       
       # Output tab
        defaultConfigs["ScreenUpdateFrequency"] = 10; paramTypeInt.append("ScreenUpdateFrequency")
        defaultConfigs["ImageOutputOn"] = False; paramTypeBool.append("ImageOutputOn")
        defaultConfigs["SaveImageFrequency"] = 100; paramTypeInt.append("SaveImageFrequency")
        defaultConfigs["Screenshot_X"] = 600; paramTypeInt.append("Screenshot_X")
        defaultConfigs["Screenshot_Y"] = 600; paramTypeInt.append("Screenshot_Y")        
        defaultConfigs["LatticeOutputOn"] = False; paramTypeBool.append("LatticeOutputOn")
        defaultConfigs["SaveLatticeFrequency"] = 100; paramTypeInt.append("SaveLatticeFrequency")
        defaultConfigs["GraphicsWinWidth"] = 400; paramTypeInt.append("GraphicsWinWidth")
        defaultConfigs["GraphicsWinHeight"] = 400; paramTypeInt.append("GraphicsWinHeight")
        defaultConfigs["UseInternalConsole"] = False; paramTypeBool.append("UseInternalConsole")
        defaultConfigs["ClosePlayerAfterSimulationDone"] = False; paramTypeBool.append("ClosePlayerAfterSimulationDone")
        
        # defaultConfigs["ProjectLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DProjects')); paramTypeString.append("ProjectLocation")
        defaultConfigs["ProjectLocation"] = QString(os.path.join(environ['PREFIX_CC3D'],'Demos')); paramTypeString.append("ProjectLocation")
        
        defaultConfigs["OutputLocation"] = QString(os.path.join(os.path.expanduser('~'),'CC3DWorkspace')); paramTypeString.append("OutputLocation")
        defaultConfigs["OutputToProjectOn"] = False; paramTypeBool.append("OutputToProjectOn")
        prefsFile = os.path.join(os.path.join(os.path.join(os.path.expanduser('~'),'.config'),ORGANIZATION),APPLICATION+'.ini')
        prefsFile = APPLICATION
        defaultConfigs["PreferencesFile"] = QString(prefsFile); paramTypeString.append("PreferencesFile")
        
        defaultConfigs["NumberOfRecentSimulations"] = 8; paramTypeInt.append("NumberOfRecentSimulations")
        
        
        # Cells/Colors tab  (used to be: Cell Type tab)
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
        defaultConfigs["WindowColorSameAsMedium"] = True; paramTypeBool.append("WindowColorSameAsMedium")        
        defaultConfigs["BrushColor"] = QColor(Qt.white); paramTypeColor.append("BrushColor")
        defaultConfigs["PenColor"] = QColor(Qt.black); paramTypeColor.append("PenColor")
        
        defaultConfigs["CellGlyphScaleByVolumeOn"] = False; paramTypeBool.append("CellGlyphScaleByVolumeOn")
        defaultConfigs["CellGlyphScale"] = 1.0; paramTypeDouble.append("CellGlyphScale")
        defaultConfigs["CellGlyphThetaRes"] = 2; paramTypeInt.append("CellGlyphThetaRes")
        defaultConfigs["CellGlyphPhiRes"] = 2; paramTypeInt.append("CellGlyphPhiRes")


        # Field tab (combines what used to be Colormap tab and Vectors tab)
        
        defaultConfigs["PixelizedScalarField"] = False; paramTypeBool.append("PixelizedScalarField")
        
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
        defaultConfigs["ScaleArrowsOn"] = False; paramTypeBool.append("ScaleArrowsOn")
        defaultConfigs["ArrowColor"] = QColor(Qt.white); paramTypeColor.append("ArrowColor")
        defaultConfigs["ArrowLength"] = 1.0; paramTypeDouble.append("ArrowLength")
        defaultConfigs["FixedArrowColorOn"] = False; paramTypeBool.append("FixedArrowColorOn")
        
        defaultConfigs["OverlayVectorsOn"] = False; paramTypeBool.append("OverlayVectorsOn")
        
        
        # 3D tab
        defaultConfigs["Types3DInvisible"] = QString("0"); paramTypeString.append("Types3DInvisible")
        defaultConfigs["BoundingBoxOn"] = True; paramTypeBool.append("BoundingBoxOn")
        defaultConfigs["BoundingBoxColor"] = QColor(Qt.white); paramTypeColor.append("BoundingBoxColor")
        
        
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
        
        defaultConfigs["FieldParams"] = {}   # NOTE!!  This needs to be last
        
        mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, APPLICATION) # use IniFormat instead of NativeFormat now
        
#        mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, ORGANIZATION, "cc3d-2")
#        initSyncSettings()
#        updatedConfigs = {}
        
        
        
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
def initFieldsParams(fieldNames):   # called from SimpleTabView once we know the fields
    print

    fieldParams = getSetting('FieldParams')

    for field in fieldNames:

        if field not in fieldParams.keys() and field != 'Cell_Field':
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
        

            fieldParams[field] = paramsDict
        

    Configuration.simFieldsParams = fieldParams

    setSetting('FieldParams',fieldParams )
    
def updateFieldsParams(fieldName,fieldDict):
    
    fieldParamsDict = getSetting("FieldParams")

    if not isinstance(fieldName,str):

        fieldName = str(fieldName)
        
    Configuration.simFieldsParams[fieldName] = fieldDict  # do regardless of in there or not
    
    setSetting('FieldParams',Configuration.simFieldsParams)
    
    
def getSetting(_key, fieldName=None):  # we append an optional fieldName now to allow for field-dependent parameters from Prefs
        if fieldName:
            if fieldName == 'Cell_Field':  # if there are no fields defined, but just the Cell_Field, return default Pref (hard-coded -> BAD)
                return getSetting(_key)
            

            fieldsDict = getSimFieldsParams()

            paramsDict = fieldsDict[fieldName]

            if _key == 'ArrowColor':  

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
        elif _key in Configuration.paramTypeBool:

            val = Configuration.mySettings.value(_key)

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

            if val.isValid():
                color = QColor(val.toString())
                return color
            else:
                color = Configuration.defaultConfigs[_key]

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
                        elif str(key2) == 'ArrowColor':  

                            dictVals[str(key2)] = str(dict2[key2].toString())
                            mycolor = QColor(dict2[key2].toString())

                        else:  dictVals[str(key2)] = dict2[key2]

                    fieldDictNew[str(key)] = dictVals
                    knt += 1

                return fieldDictNew
            else:
                fieldDict = Configuration.defaultConfigs[_key]
                
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
                
def setSetting(_key,_value):  # rf. ConfigurationDialog.py, updatePreferences()

        if _key in Configuration.paramTypeBool:
            Configuration.mySettings.setValue(_key,QVariant(_value))
        elif _key in Configuration.paramTypeInt:
            Configuration.mySettings.setValue(_key,_value)
        elif _key in Configuration.paramTypeDouble:

            Configuration.mySettings.setValue(_key,QVariant(_value))
        elif _key in Configuration.paramTypeString:

            Configuration.mySettings.setValue(_key,_value)
        elif _key in Configuration.paramTypeColor:

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
                # print 'before RecentSimulations setValue'    
                Configuration.mySettings.setValue("RecentSimulations", QVariant(recentSimulationsList))  # each time we set a list of recent files we have to update variant variable corresponding to this setting to ensure that recent file list is up to date in the GUI
                # print 'after RecentSimulations setValue'    
                

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

            if isinstance(_value,dict):

                Configuration.mySettings.setValue(_key, QVariant(_value))
            else:  # this block gets executed, it seems
                valDict = _value.toMap()
                        
                Configuration.mySettings.setValue(_key, QVariant(valDict))
            
                
            
        elif _key in ["TypeColorMap"]:

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
            

def getPlayerParams():

    playerParamsDict = {}
    for key in Configuration.defaultConfigs.keys():
        if key not in ["PlayerSizes"]:
            playerParamsDict[key] = getSetting(key)


    return playerParamsDict
    
def syncPreferences():   # this function invoked when we close the Prefs dialog with the "OK" button
    for key in Configuration.defaultConfigs.keys():
        val = Configuration.mySettings.value(key)

        if val.isValid():  # if setting exists (is valid) in the .plist
            if not key == 'RecentSimulations':
                setSetting(key,val)
        else:
            print 'setting recent simulations'
            setSetting(key,Configuration.defaultConfigs[key])
            
        
#syncPreferences()   # don't think needed; rf. UI/