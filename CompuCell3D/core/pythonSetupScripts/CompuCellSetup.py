import sys
import os
import os.path
import cStringIO, traceback
import CMLParser
from collections import OrderedDict

from distutils.dir_util import mkpath
import json

# import Configuration


cc3dModuleDictionary = {}
cc3dActiveSteerableList = []
cc3dXML2ObjConverter = None
cc3dXML2ObjConverterAdapter = None
windowsXML2ObjConverter = None
simulationXMLDescription = None
simulationObjectsCreated = False
simulationPythonScriptName = ""
simulationFileName = ""
screenshotDirectoryName = ""
customScreenshotDirectoryName = ""
invokeMethod = ""

global error_code
error_code = 0

global error_message
error_message = ''

global cml_args
cml_args = None

global cc3dSimulationDataHandler
cc3dSimulationDataHandler = None
# global extraFieldsDict
# extraFieldsDict={}

global simulation_return_value
simulation_return_value = None

global simulation_return_value_tag
simulation_return_value_tag = 'generic_tag'

global push_address
push_address = None

global current_step
current_step = 0

global simulationThreadObject
simulationThreadObject = None
global cmlParser
cmlParser = CMLParser.CMLParser()

global appendedPaths
appendedPaths = []
# PERHAPS I SHOULD ADD A CLASS WHICH WOULD CONTROL CML version

global globalSteppableRegistry  # rwh2
globalSteppableRegistry = None  # rwh2

global cmlFieldHandler
cmlFieldHandler = None
playerType = "old"
# global cc3dXML2ObjConverter_1
# cc3dXML2ObjConverter_1=None
# cc3dXML2ObjConverter_2=None

global userStopSimulationFlag
userStopSimulationFlag = False

# list of free floating SBML solvers
global freeFloatingSBMLSimulator
freeFloatingSBMLSimulator = {}  # {name:RoadRunnerPy}

global globalSBMLSimulatorOptions
globalSBMLSimulatorOptions = None  # {optionName:Value}

global viewManager
viewManager = None  # stores viewmanager object - initialized when simulation is run using Player

global steering_param_dict
steering_param_dict = OrderedDict()

global steering_panel
steering_panel = None

global steering_panel_model
steering_panel_model = None


MYMODULENAME = '------- CompuCellSetup.py: '

from enums import *


# (CELL_FIELD, CON_FIELD, SCALAR_FIELD, SCALAR_FIELD_CELL_LEVEL, VECTOR_FIELD, VECTOR_FIELD_CELL_LEVEL)=range(0,6)

class FieldRegistry:
    def __init__(self):
        self.__fieldDict = {}

    def addNewField(self, _field, _fieldName, _fieldType):
        self.__fieldDict[_fieldName] = [_field, _fieldType]

    def getFieldNames(self):
        return self.__fieldDict.keys()

    def getScalarFields(self):
        scalarFieldsDict = {}
        for fieldName in self.__fieldDict:
            if self.__fieldDict[fieldName][1] == SCALAR_FIELD:
                scalarFieldsDict[fieldName] = self.__fieldDict[fieldName][0]

        return scalarFieldsDict

    def getScalarFieldsCellLevel(self):
        scalarFieldsCellLevelDict = {}
        for fieldName in self.__fieldDict:
            if self.__fieldDict[fieldName][1] == SCALAR_FIELD_CELL_LEVEL:
                scalarFieldsCellLevelDict[fieldName] = self.__fieldDict[fieldName][0]

        return scalarFieldsCellLevelDict

    def getVectorFields(self):
        vectorFieldsDict = {}
        for fieldName in self.__fieldDict:
            if self.__fieldDict[fieldName][1] == VECTOR_FIELD:
                vectorFieldsDict[fieldName] = self.__fieldDict[fieldName][0]

        return vectorFieldsDict

    def getVectorFieldsCellLevel(self):
        vectorFieldsCellLevelDict = {}
        for fieldName in self.__fieldDict:
            if self.__fieldDict[fieldName][1] == VECTOR_FIELD_CELL_LEVEL:
                vectorFieldsCellLevelDict[fieldName] = self.__fieldDict[fieldName][0]

        return vectorFieldsCellLevelDict

    def getFieldData(self, _fieldName):
        try:
            return self.__fieldDict[_fieldName][0], self.__fieldDict[_fieldName][1]  # field, field type
        except LookupError, e:
            return None, None
        except IndexError, e:
            return None, None


global fieldRegistry
fieldRegistry = FieldRegistry()


class CustomVisData:
    def __init__(self, _visName=''):
        self.visName = _visName
        self.actorsDict = {}  # {actorName:vtk actor type - as string} - e.g. {"glyph":"vtkActor"}
        self.scriptDict = {}
        self.callbackFunction = None  # this is a function typically a member of the steppable which initializes/updates vtk actors. It is called from inside drawCustomVis function of MVCDrawView*.py classes

    def addActor(self, _actorName, _actorTypeName):
        self.actorsDict[_actorName] = _actorTypeName

    #        print MYMODULENAME,' CustomVisData:addActor()   self.actorsDict=', self.actorsDict

    def addScript(self, _fieldName, _scriptName):  # not used
        self.scriptDict[_fieldName] = _scriptName

    #        print MYMODULENAME,'  addScript(): scriptDict=',self.scriptDict

    def getActorDict(self):
        return self.actorsDict

    def getScript(self):
        return self.scriptDict

    def registerVisCallbackFunction(self, _fcn):
        self.callbackFunction = _fcn

    #        print MYMODULENAME,'  registerVisCallbackFunction(): _fcn=',_fcn

    def getVisCallbackFunction(self):
        return self.callbackFunction


class CustomVisStorage:
    def __init__(self):
        self.visDataDict = {}  # {visName:visData}
        # self.callbackFunctionDict={}   #{visName:visCallBackFunction_steppable}

    def addNewVisData(self, _visName):
        self.visDataDict[_visName] = CustomVisData(_visName)

    def getVisData(self, _visName):
        try:
            return self.visDataDict[_visName]
        except LookupError, e:
            return None


global customVisStorage
customVisStorage = CustomVisStorage()


class CC3DCPlusPlusError(Exception):
    def __init__(self, _message):
        self.message = _message

    def __str__(self):
        return repr(self.message)


# Class that handles all troubles with XML and Python files consistencies
class SimulationPaths:
    def __init__(self):
        self.lock = False

        self.playerSimulationXMLFileName = ""
        self.playerSimulationPythonScriptName = ""

        self.playerSimulationXMLFilePath = ""
        self.playerSimulationPythonScriptPath = ""

        self.pythonScriptNameFromXML = ""
        self.pathToPythonScriptNameFromXML = ""

        self.xmlFileNameFromPython = ""
        self.pathToxmlFileNameFromPython = ""

        self.playerSimulationWindowsFileName = ""  # xml file describing layout of graphics windows in player (not plotting windows)
        self.playerSimulationWindowsFilePath = ""

        # self.mainSimulationFile="" # this is the name of the main simulation file. It may contain references to other files. This is the file that is input from CML or from Player 
        self.simulationResultStorageDirectory = ""  # this is a place where the output of simulation will be written
        self.simulationResultDescriptionFile = ""  # this is a place where the output of simulation is stored and where LDS file is stored

        # these two variables store XML and python files that are actually used in the simulation
        self.simulationXMLFileName = ""
        self.simulationPythonScriptName = ""

        self.basePath = ""

    def setSimulationBasePath(self, _path):
        self.basePath = os.path.abspath(_path)

    def setSimulationResultDescriptionFile(self, _fileName):
        self.simulationResultDescriptionFile = os.path.abspath(_fileName)
        self.simulationResultStorageDirectory = os.path.dirname(self.simulationResultDescriptionFile)

    def getSimulationResultStorageDirectory(self):
        return self.simulationResultStorageDirectory

    def setSimulationResultStorageDirectory(self, _path, _useTimeExtension=False):
        mainDirName = os.path.dirname(os.path.abspath(_path))
        baseName = os.path.basename(os.path.abspath(_path))
        baseName = baseName.replace('.', '_')
        if not _useTimeExtension:
            self.simulationResultStorageDirectory = os.path.join(mainDirName, baseName)
            global screenshotDirectoryName
            screenshotDirectoryName = self.simulationResultStorageDirectory
            print MYMODULENAME, "screenshotDirectoryName=", screenshotDirectoryName
        else:
            import time
            timeNameExtension = "_" + time.strftime("%m", time.localtime()) + "_" + time.strftime("%d",
                                                                                                  time.localtime()) + "_" + time.strftime(
                "%Y", time.localtime()) \
                                + "_" + time.strftime("%H", time.localtime()) + "_" + time.strftime("%M",
                                                                                                    time.localtime()) + "_" + time.strftime(
                "%S", time.localtime())

            self.simulationResultStorageDirectory = os.path.join(mainDirName, baseName + timeNameExtension)
            global screenshotDirectoryName
            screenshotDirectoryName = self.simulationResultStorageDirectory
            #            print MYMODULENAME,' -----------    appending timestamp suffix:  screenshotDirectoryName=',screenshotDirectoryName

    def setSimulationResultStorageDirectoryDirect(self, _path, ):
        self.simulationResultStorageDirectory = _path
        global screenshotDirectoryName
        screenshotDirectoryName = self.simulationResultStorageDirectory

    def getFullFileNameAndFilePath(self, _fileName, _filePath):
        if _fileName == "":
            return "", ""
        fileName = os.path.abspath(_fileName)
        filePath = os.path.dirname(_fileName)
        return fileName, filePath

    def setPlayerSimulationXMLFileName(self, _playerSimulationXMLFileName):
        if self.lock:
            return
        self.playerSimulationXMLFileName = _playerSimulationXMLFileName
        self.playerSimulationXMLFileName = self.normalizePath(self.playerSimulationXMLFileName)
        self.playerSimulationXMLFileName, self.playerSimulationXMLFilePath = self.getFullFileNameAndFilePath(
            self.playerSimulationXMLFileName, self.playerSimulationXMLFilePath)

        if self.playerSimulationXMLFileName != "":
            assert os.path.exists(
                self.playerSimulationXMLFileName), "In the Player you requested XML file: \n" + self.playerSimulationXMLFileName \
                                                   + "\nwhich does not exist.\nPlease pick different XML file "

        self.simulationXMLFileName = self.playerSimulationXMLFileName

    def setPlayerSimulationPythonScriptName(self, _playerSimulationPythonScriptName):
        if self.lock:
            return
        self.playerSimulationPythonScriptName = _playerSimulationPythonScriptName
        self.playerSimulationPythonScriptName = self.normalizePath(self.playerSimulationPythonScriptName)
        self.playerSimulationPythonScriptName, self.playerSimulationPythonScriptPath = self.getFullFileNameAndFilePath(
            self.playerSimulationPythonScriptName, self.playerSimulationPythonScriptPath)

        if self.playerSimulationPythonScriptName != "":
            assert os.path.exists(
                self.playerSimulationPythonScriptName), "In the Player you requested Python script: \n" \
                                                        + self.playerSimulationPythonScriptName + "\nwhich does not exist.\nPlease pick different script"

        self.simulationPythonScriptName = self.playerSimulationPythonScriptName

    def setPlayerSimulationWindowsFileName(self, _filename):
        if self.lock:
            return
        self.playerSimulationWindowsFileName = _filename
        self.playerSimulationWindowsFileName = self.normalizePath(self.playerSimulationWindowsFileName)
        self.playerSimulationWindowsFileName, self.playerSimulationWindowsFilePath = self.getFullFileNameAndFilePath(
            self.playerSimulationWindowsFileName, self.playerSimulationWindowsFilePath)
        print MYMODULENAME, "self.playerSimulationWindowsFileName=", self.playerSimulationWindowsFileName

        if self.playerSimulationWindowsFileName != "":
            assert os.path.exists(
                self.playerSimulationWindowsFileName), "In the Player you requested Windows XML file: \n" + self.playerSimulationWindowsFileName \
                                                       + "\nwhich does not exist.\nPlease pick different Windows XML file "

        self.simulationWindowsFileName = self.playerSimulationWindowsFileName

    def setPythonScriptNameFromXML(self, _pythonScriptNameFromXML):
        if self.lock:
            return
        self.pythonScriptNameFromXML = _pythonScriptNameFromXML
        self.pythonScriptNameFromXML = self.normalizePath(self.pythonScriptNameFromXML)
        self.pythonScriptNameFromXML, self.pathToPythonScriptNameFromXML = self.getFullFileNameAndFilePath(
            self.pythonScriptNameFromXML, self.pathToPythonScriptNameFromXML)

        print "_pythonScriptNameFromXML=", _pythonScriptNameFromXML
        self.pythonScriptNameFromXML = self.normalizePath(_pythonScriptNameFromXML)
        print "_pythonScriptNameFromXML=", self.pythonScriptNameFromXML
        print "self.basePath=", self.basePath
        # sys.exit()

        if self.simulationPythonScriptName == "":
            assert os.path.exists(
                self.pythonScriptNameFromXML), "In the XML file you have specified Python script: \n" + self.pythonScriptNameFromXML \
                                               + "  which does not exist.\n Please specify different Python script"
            self.simulationPythonScriptName = self.pythonScriptNameFromXML

            # try:
            # open(_pythonScriptNameFromXML)
            # except:
            # print " THE FILE ",_pythonScriptNameFromXML," does not exist"
            # sys.exit()            

    def setXmlFileNameFromPython(self, _xmlFileNameFromPython):
        if self.lock:
            return
        self.xmlFileNameFromPython = _xmlFileNameFromPython
        self.xmlFileNameFromPython = self.normalizePath(self.xmlFileNameFromPython)
        self.xmlFileNameFromPython, self.pathToxmlFileNameFromPython = self.getFullFileNameAndFilePath(
            self.xmlFileNameFromPython, self.pathToxmlFileNameFromPython)

        if self.simulationXMLFileName == "":
            assert os.path.exists(
                self.xmlFileNameFromPython), "In the Python script you requested XML file: \n" + self.xmlFileNameFromPython \
                                             + "  which does not exist.\n Please specify different XML file"
            self.simulationXMLFileName = self.xmlFileNameFromPython;

    def normalizePath(self, _path):
        # print "INSIDE NORMALIZE PATH _arg=",_path
        if self.basePath == "":
            return os.path.abspath(_path)

        path = _path
        # print "TRY OPENING FILES"
        try:
            # first we try if the path exists
            open(_path)
            # print "SUCCESFULLY OPENED ",_path    
            return os.path.abspath(_path)
        except:
            # print "EXCEPTION"
            # if it does not we try to see if combination of _basePath/path exists - in this case _path may be relative path 
            try:
                # print "TRYING DIFFERENT PATH"
                path = os.path.join(self.basePath, _path)
                # print "ATTEMPT NORMALIZAED PATH=",os.path.abspath(path)
                open(path)
                # print "NORMALIZAED PATH=",os.path.abspath(path)
                return os.path.abspath(path)
            except:
                return _path

        return _path

    def ensurePathsConsistency(self):
        self.lock = True

        if self.basePath != "":
            # we will normalize all paths before doing any comparison
            print "\n\n\n\n self.basePath=", self.basePath
            print "self.playerSimulationXMLFileName=", self.playerSimulationXMLFileName
            if self.playerSimulationXMLFileName != "":
                self.playerSimulationXMLFileName = self.normalizePath(self.playerSimulationXMLFileName)

            if self.xmlFileNameFromPython != "":
                self.xmlFileNameFromPython = self.normalizePath(self.xmlFileNameFromPython)

            if self.playerSimulationPythonScriptName != "":
                self.playerSimulationPythonScriptName = self.normalizePath(self.playerSimulationPythonScriptName)

            if self.pythonScriptNameFromXML != "":
                self.pythonScriptNameFromXML = self.normalizePath(self.pythonScriptNameFromXML)

        if self.playerSimulationXMLFileName != "" and self.xmlFileNameFromPython == "":
            self.simulationXMLFileName = self.playerSimulationXMLFileName
        if self.playerSimulationXMLFileName == "" and self.xmlFileNameFromPython != "":
            self.simulationXMLFileName = self.xmlFileNameFromPython
        if self.playerSimulationXMLFileName != "" and self.xmlFileNameFromPython != "":
            assert self.playerSimulationXMLFileName == self.xmlFileNameFromPython, "XML file specified in the player: \n" + self.playerSimulationXMLFileName \
                                                                                   + "\nis different from XML file set in the Python script: \n" + self.xmlFileNameFromPython \
                                                                                   + "\nPlease make sure the two XML files are the same\n" \
                                                                                   + "You may also try removing line in the Python script where you set XML file name or unselect XML file in the Player"
            self.simulationXMLFileName = self.xmlFileNameFromPython

        if self.playerSimulationPythonScriptName != "" and self.pythonScriptNameFromXML == "":
            self.simulationPythonScriptName = self.playerSimulationPythonScriptName

        if self.playerSimulationPythonScriptName == "" and self.pythonScriptNameFromXML != "":
            self.simulationPythonScriptName = self.pythonScriptNameFromXML
        if self.playerSimulationPythonScriptName != "" and self.pythonScriptNameFromXML != "":
            assert self.playerSimulationPythonScriptName == self.pythonScriptNameFromXML, "Python script from Player: \n" + self.playerSimulationPythonScriptName \
                                                                                          + "\nis different than Python script specified in XML: \n" + self.pythonScriptNameFromXML \
                                                                                          + "\nPlease make sure the two scripts are the same\n" \
                                                                                          + "You may also try removing <PythonScript> tag from XML or unselecting Python script in the Player"
            self.simulationPythonScriptName = self.playerSimulationPythonScriptName

        global simulationPythonScriptName
        simulationPythonScriptName = self.simulationPythonScriptName

        simulationFileName = self.simulationXMLFileName

        print "simulationPythonScriptName=", simulationPythonScriptName, " simulationFileName=", simulationFileName
        assert not (
            simulationFileName == "" and simulationPythonScriptName == ""), "You have not specified any simulation file"


simulationPaths = SimulationPaths()  # will store simulation file names and paths and check if paths are correct


def resetGlobals():
    global simulationObjectsCreated
    simulationObjectsCreated = False
    global simulationFileName
    simulationFileName = ""
    global screenshotDirectoryName
    screenshotDirectoryName = ""

    global customScreenshotDirectoryName
    customScreenshotDirectoryName = ''

    global cc3dSimulationDataHandler
    cc3dSimulationDataHandler = None

    global fieldRegistry
    fieldRegistry = FieldRegistry()

    global simulationThreadObject
    simulationThreadObject = None

    global appendedPaths
    for path in appendedPaths:
        # removing all occurences
        sys.path.remove(path)

    appendedPaths = []

    global freeFloatingSBMLSimulator
    freeFloatingSBMLSimulator = {}

    global globalSBMLSimulatorOptions
    globalSBMLSimulatorOptions = None

    global steering_param_dict
    steering_param_dict = OrderedDict()

    global steering_panel
    steering_panel = None

    global steering_panel_model
    steering_panel_model = None



def setSimulationXMLFileName(_simulationFileName):
    global simulationPaths
    simulationPaths.setXmlFileNameFromPython(_simulationFileName)


#     print "\n\n\n got here ",simulationPaths.simulationXMLFileName


def addSteeringPanel(panel_data):
    """
    Adds steering panel with sliders to the Player5 Window
    :return:
    """
    global viewManager
    # global steering_panel
    # global steering_panel_model

    steering_panel = viewManager.widgetManager.getNewWidget('Steering Panel', panel_data)
    return steering_panel

def serialize_steering_panel(fname):
    """
    Serializes steering panel
    :param fname: {str} filename - to serialize steering panel to
    :return: None
    """
    global steering_param_dict
    json_dict = OrderedDict()

    for param_name, steering_param_obj in steering_param_dict.items():
        json_dict[param_name] = OrderedDict()
        json_dict[param_name]['val'] = steering_param_obj.val
        json_dict[param_name]['min'] = steering_param_obj.min
        json_dict[param_name]['max'] = steering_param_obj.max
        json_dict[param_name]['decimal_precision'] = steering_param_obj.decimal_precision
        json_dict[param_name]['enum'] = steering_param_obj.enum
        json_dict[param_name]['widget_name'] = steering_param_obj.widget_name


    with open(fname, 'w') as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)


def deserialize_steering_panel(fname):
    """
    Deserializes steering panel
    :param fname: {str} serialized steering panel filename
    :return: None
    """

    global steering_param_dict
    import SteeringParam

    with open(fname) as infile:

        json_dict = json.load(infile)
        steering_param_dict = OrderedDict()
        for param_name, steering_param_obj_data in json_dict.items():
            try:
                steering_param_dict[param_name] = SteeringParam.SteeringParam(
                    name=param_name,
                    val=steering_param_obj_data['val'],
                    min_val=steering_param_obj_data['min'],
                    max_val=steering_param_obj_data['max'],
                    decimal_precision=steering_param_obj_data['decimal_precision'],
                    enum=steering_param_obj_data['enum'],
                    widget_name=steering_param_obj_data['widget_name']
                )
            except:
                print 'Could not update steering parameter {} value'.format(param_name)


def addNewPlotWindow(_title='', _xAxisTitle='', _yAxisTitle='', _xScaleType='linear', _yScaleType='linear', _grid=True,_config_options=None):
    class PlotWindowDummy(object):
        '''
        This class serves as a dummy object that is used when viewManager is None
        It "emulates" plotWindow API but does nothing except ensures that
        in the simulation that does not use CC3D Player (ciew manager in such simulation is set to None)
        any call to plotWindow object gets captured/intercepted by __getattr__ function which returns a dummy method that accepts any
        type of arguments and has ampty body. This way any calls to plots defined in the simulation that runs in the console are gently
        ignored and simulation runs just fine. Most importantly it does not require plots to be commented out int he command line applications
        '''

        def __init__(self):
            pass

        def __getattr__(self, attr):
            return self.dummyFcn

        def dummyFcn(self, *args, **kwds):
            pass

    global viewManager

    if not viewManager:
        pwd = PlotWindowDummy()
        return pwd

    qt_version = 4
    try:
        pW = viewManager.plotManager.getNewPlotWindow()
    except:
        full_options_dict = {
            'title':_title,
            'xAxisTitle':_xAxisTitle,
            'yAxisTitle':_yAxisTitle,
            'xScaleType':_xScaleType,
            'yScaleType':_yScaleType,
            'grid':_grid
        }
        if _config_options:
            full_options_dict.update(_config_options)

        pW = viewManager.plotManager.getNewPlotWindow(full_options_dict)

        # pW = viewManager.plotManager.getNewPlotWindow({
        #     'title':_title,
        #     'xAxisTitle':_xAxisTitle,
        #     'yAxisTitle':_yAxisTitle,
        #     'xScaleType':_xScaleType,
        #     'yScaleType':_yScaleType,
        #     'grid':_grid
        # }
        # )
        qt_version = 5

    if not pW:
        raise AttributeError(
            'Missing plot modules. Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads. Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages')

    # setting up default plot window parameters/look
    if qt_version==4:
        # Plot Title - properties
        pW.setTitle(_title)
        pW.setTitleSize(12)
        pW.setTitleColor("Green")

        # plot background
        pW.setPlotBackgroundColor("white")
        # properties of x axis
        pW.setXAxisTitle(_xAxisTitle)
        if _xScaleType == 'log':
            pW.setXAxisLogScale()
        pW.setXAxisTitleSize(10)
        pW.setXAxisTitleColor("blue")

        # properties of y axis
        pW.setYAxisTitle(_yAxisTitle)
        if _xScaleType == 'log':
            pW.setYAxisLogScale()
        pW.setYAxisTitleSize(10)
        pW.setYAxisTitleColor("red")

        pW.addGrid()
        # adding automatically generated legend
        # default possition is at the bottom of the plot but here we put it at the top
        pW.addAutoLegend("top")

        # restoring plot window - have to decide whether to keep it or rely on viewManager.plotManager restore_plots_layout function
        #     viewManager.plotManager.restoreSingleWindow(pW)

    return pW


def makeDefaultSimulationOutputDirectory():
    global simulationFileName
    global screenshotDirectoryName

    simulationFileBaseName = os.path.basename(simulationFileName)
    import datetime
    timestampStr = datetime.datetime.now().__str__()
    timestampStr = timestampStr.replace(' ', '-')
    timestampStr = timestampStr.replace('.', '-')
    timestampStr = timestampStr.replace(':', '-')

    defaultOutputRootDirectory = os.path.join(os.path.expanduser('~'), 'CC3DWorkspace')
    outputDir = os.path.join(defaultOutputRootDirectory, simulationFileBaseName + '-' + timestampStr)
    outputDir = os.path.abspath(outputDir)  # normalizing path

    screenshotDirectoryName = outputDir  # after this call screenshot directory  - screenshotDirectoryName is set to outputDir

    try:
        mkpath(outputDir)
        return outputDir
    except:
        raise IOError


def getSimulationOutputDir():
    global screenshotDirectoryName
    if screenshotDirectoryName == '':
        return makeDefaultSimulationOutputDirectory()
    else:
        return os.path.abspath(screenshotDirectoryName)  # normalizing path


def openFileInSimulationOutputDirectory(_filePath, _mode="r"):
    global screenshotDirectoryName

    # print 'screenshotDirectoryName=',screenshotDirectoryName
    #     print 'dirName=',dirName
    #     print 'fileName=',fileName

    if screenshotDirectoryName == "":
        fileName = os.path.abspath(_filePath)
        dirName = os.path.dirname(fileName)

        defaultOutputRootDirectory = makeDefaultSimulationOutputDirectory()

        outputFileName = os.path.join(defaultOutputRootDirectory, _filePath)
        outputFileName = os.path.abspath(outputFileName)  # normalizing path
        dirForOutputFileName = os.path.dirname(outputFileName)

        try:
            mkpath(dirForOutputFileName)
        except:
            raise IOError

        try:
            return open(outputFileName, _mode), outputFileName
        except:
            raise IOError('COULD NOT OPEN ' + outputFileName + ' in mode=' + _mode)

    fileName = os.path.join(os.path.abspath(screenshotDirectoryName), _filePath)
    fileName = os.path.abspath(fileName)  # normalizing path

    dirName = os.path.dirname(fileName)

    if os.path.abspath(screenshotDirectoryName) == dirName:
        try:
            return open(fileName, _mode), fileName
        except:  # perhaps directory screenshotDirectoryName does not exist yet
            try:
                mkpath(dirName)
                return open(fileName, _mode), fileName
            except:

                raise IOError
    else:
        try:
            mkpath(dirName)
        except:
            raise IOError

        try:
            return open(fileName, _mode), fileName
        except:
            raise IOError

    raise IOError


# Load data for Plugins, Steppables and Potts
def initModules(sim, _cc3dXML2ObjConverter):
    import XMLUtils
    #     global cc3dXML2ObjConverter

    pluginDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Plugin"))
    for pluginData in pluginDataList:
        print "Element", pluginData.name
        sim.ps.addPluginDataCC3D(pluginData)

    steppableDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Steppable"))
    for steppableData in steppableDataList:
        print "Element", steppableData.name
        sim.ps.addSteppableDataCC3D(steppableData)

    pottsDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Potts"))
    assert pottsDataList.getBaseClass().size() <= 1, 'You have more than 1 definition of the Potts section'
    if pottsDataList.getBaseClass().size() == 1:
        for pottsData in pottsDataList:
            print "Element", pottsData.name
            sim.ps.addPottsDataCC3D(pottsData)

    metadataDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Metadata"))
    assert metadataDataList.getBaseClass().size() <= 1, 'You have more than 1 definition of the Metadata section'
    if metadataDataList.getBaseClass().size() == 1:
        for metadataData in metadataDataList:
            print "Element", metadataData.name
            sim.ps.addMetadataDataCC3D(metadataData)


# print "\n\n\n\n\n ITERATION"
#     walker=XMLUtils.CC3DXMLElementWalker()
#     walker.iterateCC3DXMLElement(cc3dXML2ObjConverter.root)

def parseXML(_simulationFileName):
    global cc3dXML2ObjConverter
    import XMLUtils
    cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
    root_element = cc3dXML2ObjConverter.Parse(_simulationFileName)


def parseWindowsXML(_windowsXMLFileName):
    global windowsXML2ObjConverter
    import XMLUtils
    windowsXML2ObjConverter = XMLUtils.Xml2Obj()
    root_element = windowsXML2ObjConverter.Parse(_windowsXMLFileName)


    # global cc3dXML2ObjConverter_1
    # cc3dXML2ObjConverter_1=XMLUtils.Xml2Obj()
    # root_element_1=cc3dXML2ObjConverter_1.Parse(_simulationFileName)    
    # import time
    # time.sleep(2)


    # text = raw_input(' PARSING XML->')

    # print "PARSED cc3dXML2ObjConverter_1"
    # print "cc3dXML2ObjConverter_1.root=",cc3dXML2ObjConverter_1.root


class XML2ObjConverterAdapter:
    def __init__(self):
        self.root = None
        self.xmlTree = None


def setSimulationXMLDescription(_xmlTree):
    if playerType == "CML":
        setSimulationXMLDescriptionNewPlayer(_xmlTree)
    else:
        if simulationThreadObject is None:
            print MYMODULENAME, '  setSimulationXMLDescription():  error, simThreadObject is None'
        # setSimulationXMLDescriptionOldPlayer(_xmlTree)
        else:
            setSimulationXMLDescriptionNewPlayer(_xmlTree)
            # global cc3dXML2ObjConverterAdapter
            # cc3dXML2ObjConverterAdapter=XML2ObjConverterAdapter()
            # cc3dXML2ObjConverterAdapter.xmlTree=_xmlTree
            # cc3dXML2ObjConverterAdapter.root=_xmlTree.CC3DXMLElement


def setSimulationXMLDescriptionNewPlayer(_xmlTree):
    global cc3dXML2ObjConverter
    cc3dXML2ObjConverter = XML2ObjConverterAdapter()
    cc3dXML2ObjConverter.xmlTree = _xmlTree
    cc3dXML2ObjConverter.root = _xmlTree.CC3DXMLElement


def getScreenshotDirectoryName():
    global screenshotDirectoryName
    return screenshotDirectoryName


def getCoreSimulationObjectsNewPlayer(_parseOnlyFlag=False, _cmlOnly=False):
    import sys
    from os import environ
    import string
    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)

        # sys.path.append(environ["PYTHON_MODULE_PATH"])

    import SystemUtils
    SystemUtils.setSwigPaths()
    SystemUtils.initializeSystemResources()
    # this dummy library was necessary to get restarting of the Python interpreter from C++ to work with SWIG generated libraries
    import Example

    import CompuCell
    CompuCell.initializePlugins()
    simthread = None
    sim = None

    # Here I will extract file names from the Player . Note that this is done only if the _parseOnlyFlag is set i.e. when we determine which files to use
    # if _parseOnlyFlag:
    # global simulationPaths

    # if simthread is not None:
    # xmlfile = simthread.getSimulationFileName()
    # script  = simthread.getSimulationPythonScriptName()
    # else:
    # xmlfile = filename
    # script  = ""

    # simulationPaths.setPlayerSimulationXMLFileName(xmlfile)
    # simulationPaths.setPlayerSimulationPythonScriptName(script)

    # if simulationPaths.simulationXMLFileName!="" and _parseOnlyFlag:

    # global cc3dXML2ObjConverter
    # import XMLUtils
    # parseXML(simulationPaths.simulationXMLFileName)

    # # will try extracting python script name from simulation description if the one from Player is = ""


    # global simulationPaths
    # if cc3dXML2ObjConverter.root.findElement("PythonScript"):
    # simulationPaths.setPythonScriptNameFromXML(cc3dXML2ObjConverter.root.getFirstElement("PythonScript").getText())

    global simulationPaths
    if not _parseOnlyFlag:
        sim = CompuCell.Simulator()
        sim.setNewPlayerFlag(True)
        sim.setBasePath(simulationPaths.basePath)
        if simthread is not None:
            simthread.setSimulator(sim)

        if simulationPaths.simulationXMLFileName != "":
            global simulationPaths
            global cc3dXML2ObjConverter
            import XMLUtils

            parseXML(simulationPaths.simulationXMLFileName)

            if cc3dXML2ObjConverter.root.findElement("PythonScript"):
                simulationPaths.setPythonScriptNameFromXML(
                    cc3dXML2ObjConverter.root.getFirstElement("PythonScript").getText())

        #simulationPaths.ensurePathsConsistency()

        # #here I will append path to search paths based on the paths to XML file and Python script paths
        # if simulationPaths.playerSimulationPythonScriptPath != "":
        # sys.path.append(simulationPaths.playerSimulationPythonScriptPath)

        # if simulationPaths.pathToPythonScriptNameFromXML !="":
        # sys.path.append(simulationPaths.pathToPythonScriptNameFromXML)

        # if simulationPaths.playerSimulationXMLFilePath !="":
        # sys.path.append(simulationPaths.playerSimulationXMLFilePath)

        # if simulationPaths.pathToxmlFileNameFromPython!="":
        # sys.path.append(simulationPaths.pathToxmlFileNameFromPython)


        # here I will append path to search paths based on the paths to XML file and Python script paths
        global appendedPaths
        if simulationPaths.playerSimulationPythonScriptPath != "":
            sys.path.insert(0, simulationPaths.playerSimulationPythonScriptPath)
            appendedPaths.append(simulationPaths.playerSimulationPythonScriptPath)

        if simulationPaths.pathToPythonScriptNameFromXML != "":
            sys.path.insert(0, simulationPaths.pathToPythonScriptNameFromXML)
            appendedPaths.append(simulationPaths.pathToPythonScriptNameFromXML)

        if simulationPaths.playerSimulationXMLFilePath != "":
            sys.path.insert(0, simulationPaths.playerSimulationXMLFilePath)
            appendedPaths.append(simulationPaths.playerSimulationXMLFilePath)

        if simulationPaths.pathToxmlFileNameFromPython != "":
            sys.path.insert(0, simulationPaths.pathToxmlFileNameFromPython)
            appendedPaths.append(simulationPaths.pathToxmlFileNameFromPython)

        # initModules(sim)#extracts Plugins, Steppables and Potts XML elements and passes it to the simulator


        global simulationObjectsCreated
        simulationObjectsCreated = True

    if not _cmlOnly:
        # sys.exit()        
        global simulationThreadObject

        # simulationThreadObject.sim = sim
        simulationThreadObject.setSimulator(sim)

        return sim, simulationThreadObject

    else:
        global cmlFieldHandler
        # import CMLFieldHandler        
        # cmlFieldHandler=CMLFieldHandler.CMLFieldHandler()
        # cmlFieldHandler.sim=sim
        createCMLFileHandler(sim)
        return sim, cmlFieldHandler


def createCMLFileHandler(sim):
    global cmlFieldHandler  # rwh2
    import CMLFieldHandler
    #    print MYMODULENAME,' -------- createCMLFileHandler called, sim=',sim
    cmlFieldHandler = CMLFieldHandler.CMLFieldHandler()
    cmlFieldHandler.sim = sim


#    return cmlFieldHandler   #rwh

def getCoreSimulationObjectsNewPlayerCMLReplay(_parseOnlyFlag=False, _cmlOnly=False):
    import sys
    from os import environ
    import string
    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)

        # sys.path.append(environ["PYTHON_MODULE_PATH"])

    import SystemUtils
    SystemUtils.setSwigPaths()
    SystemUtils.initializeSystemResources()
    # this dummy library was necessary to get restarting of the Python interpreter from C++ to work with SWIG generated libraries
    import Example

    import CompuCell
    # CompuCell.initializePlugins()
    simthread = None
    sim = None

    # global cmlFieldHandler
    # import CMLResultReader
    # cmlFieldHandler=CMLResultReader.CMLResultReader()
    # return sim,cmlFieldHandler    

    return sim, simulationThreadObject


def getCoreSimulationObjects(_parseOnlyFlag=False):
    global playerType
    #    print MYMODULENAME, 'getCoreSimulationObjects: playerType=',playerType
    if playerType == "CML":
        return getCoreSimulationObjectsNewPlayer(_parseOnlyFlag, True)

    if playerType == "CMLResultReplay":
        return getCoreSimulationObjectsNewPlayerCMLReplay(_parseOnlyFlag)

    if simulationThreadObject is None:
        #        return getCoreSimulationObjectsOldPlayer(_parseOnlyFlag)
        print MYMODULENAME, ' getCoreSimulationObjects(): error, simThreadObj is None'
    else:
        return getCoreSimulationObjectsNewPlayer(_parseOnlyFlag)


def createCMLFieldHandler():
    global cmlFieldHandler
    import CMLFieldHandler

    cmlFieldHandler = CMLFieldHandler.CMLFieldHandler()
    # cmlFieldHandler.sim=sim


#    print MYMODULENAME,"createCMLFieldHandler():  created cmlFieldHandler=",cmlFieldHandler

#    cmlFieldHandler.sim = sim   # rwh


def initCMLFieldHandler(sim, simulationResultStorageDirectory, _fieldStorage):
    global cmlFieldHandler  # rwh2
    #    print MYMODULENAME,'initCMLFieldHandler():'
    #    if not cmlFieldHandler:   #rwh
    #        print MYMODULENAME,' initCMLFieldHandler(), cmlFieldHandler is null'
    #        createCMLFieldHandler()
    #        cmlFieldHandler.sim = sim
    ##        createCMLFileHandler()
    #        cmlFieldHandler.getInfoAboutFields()
    #    else:
    #        cmlFieldHandler.sim = sim
    cmlFieldHandler.sim = sim

    # print "assignedSimulator object sim= ",cmlFieldHandler.sim
    # sys.exit()
    cmlFieldHandler.fieldWriter.init(sim)
    cmlFieldHandler.setFieldStorage(_fieldStorage)
    # cmlFieldHandler.getInfoAboutFields()
    # cmlFieldHandler.outputFrequency=cmlParser.outputFrequency
    # cmlFieldHandler.outputFileCoreName=cmlParser.outputFileCoreName

    cmlFieldHandler.prepareSimulationStorageDir(simulationResultStorageDirectory)
    #    print MYMODULENAME,' initCMLFieldHandler(),  calling setMaxNumberOfSteps, sim.getNumSteps()=',sim.getNumSteps()
    cmlFieldHandler.setMaxNumberOfSteps(
        sim.getNumSteps())  # will determine the length text field  of the step number suffix
    # cmlFieldHandler.writeXMLDescriptionFile() # initialization of the cmlFieldHandler is done - we can write XML description file                     

    # sys.exit()


def ExtractPythonScriptNameFromXML(_simulationXMLFileName):
    import XMLUtils
    cc3dXML2ObjConverterTmp = XMLUtils.Xml2Obj()
    root_element_tmp = cc3dXML2ObjConverterTmp.Parse(_simulationXMLFileName)

    # parseXML(_simulationXMLFileName)
    # will try extracting python script name from simulation description if the one from Player is = ""

    global simulationPaths
    if cc3dXML2ObjConverterTmp.root.findElement("PythonScript"):
        pythonScriptName = simulationPaths.normalizePath(
            cc3dXML2ObjConverterTmp.root.getFirstElement("PythonScript").getText())
        # pythonScriptName=os.path.abspath(cc3dXML2ObjConverterTmp.root.getFirstElement("PythonScript").getText())
        return pythonScriptName
    else:
        return ""


def clearActiveSteerableList(_list):
    while len(_list):
        _list.pop()


def getModuleParseData(moduleName):
    for key in cc3dModuleDictionary.keys():
        print MYMODULENAME, "This a name of a module: ", cc3dModuleDictionary[key].moduleName
    return cc3dModuleDictionary[moduleName]


def steer(_parseData):
    cc3dActiveSteerableList.append(_parseData)


def executeSteering(_sim):
    for pd in cc3dActiveSteerableList:
        module = _sim.getSteerableObject(pd.moduleName)
        module.update(pd)
    clearActiveSteerableList(cc3dActiveSteerableList)


def registerPlugin(_sim, _pd):
    _sim.ps.addPluginData(_pd)
    #     _sim.ps.pluginParseDataVector.push_back(_pd)
    cc3dModuleDictionary[_pd.moduleName] = _pd


def registerSteppable(_sim, _pd):
    _sim.ps.addSteppableData(_pd)
    #     _sim.ps.steppableParseDataVector.push_back(_pd)
    cc3dModuleDictionary[_pd.moduleName] = _pd


def registerPotts(_sim, _pd):
    _sim.ps.addPottsData(_pd)
    cc3dModuleDictionary[_pd.moduleName] = _pd


def getSteppablesByClassName(_className):
    '''
    This function returns a list of registered steppables with class name given  by _className
    '''
    global globalSteppableRegistry
    globalSteppableRegistry.getSteppablesByClassName(_className)


def getSteppableRegistry():
    from PySteppables import SteppableRegistry
    steppableRegistry = SteppableRegistry()
    return steppableRegistry


def getEnergyFunctionRegistry(sim):
    import CompuCell
    energyFunctionRegistry = CompuCell.EnergyFunctionPyWrapper()
    energyFunctionRegistry.setSimulator(sim)
    energyFunctionRegistry.setPotts(sim.getPotts())
    sim.getPotts().registerEnergyFunction(energyFunctionRegistry.getEnergyFunctionPyWrapperPtr())
    return energyFunctionRegistry


def getChangeWatcherRegistry(sim):
    import CompuCell
    changeWatcherRegistry = CompuCell.ChangeWatcherPyWrapper()
    changeWatcherRegistry.setSimulator(sim)
    changeWatcherRegistry.setPotts(sim.getPotts())
    sim.getPotts().registerCellGChangeWatcher(changeWatcherRegistry.getChangeWatcherPyWrapperPtr())
    return changeWatcherRegistry


def getStepperRegistry(sim):
    import CompuCell
    stepperRegistry = CompuCell.StepperPyWrapper()
    stepperRegistry.setSimulator(sim)
    stepperRegistry.setPotts(sim.getPotts())
    sim.getPotts().registerStepper(stepperRegistry.getStepperPyWrapperPtr())
    return stepperRegistry


def ExtractLatticeType():
    global cc3dXML2ObjConverter
    if cc3dXML2ObjConverter.root.findElement("Potts"):
        if cc3dXML2ObjConverter.root.getFirstElement("Potts").findElement("LatticeType"):
            return cc3dXML2ObjConverter.root.getFirstElement("Potts").getFirstElement("LatticeType").getText()

    return ""


def ExtractTypeNamesAndIds():
    global cc3dXML2ObjConverter
    if cc3dXML2ObjConverter is None:
        return

    pluginElements = cc3dXML2ObjConverter.root.getElements("Plugin")

    from XMLUtils import CC3DXMLListPy
    listPlugin = CC3DXMLListPy(pluginElements)
    typeIdTypeNameDict = {}
    for element in listPlugin:

        if element.getAttribute("Name") == "CellType":
            cellTypesElements = element.getElements("CellType")

            listCellTypeElements = CC3DXMLListPy(cellTypesElements)
            for cellTypeElement in listCellTypeElements:
                typeName = ""
                typeId = 0
                typeName = cellTypeElement.getAttribute("TypeName")
                typeId = cellTypeElement.getAttributeAsInt("TypeId")
                typeIdTypeNameDict[typeId] = typeName

    return typeIdTypeNameDict


def initializeSimulationObjects(sim, simthread):
    if not playerType == "CMLResultReplay":
        global cc3dXML2ObjConverter
        global cc3dXML2ObjConverterAdapter
        if cc3dXML2ObjConverter is not None:
            initModules(sim,
                        cc3dXML2ObjConverter)  # extracts Plugins, Steppables and Potts XML elements and passes it to the simulator
        if cc3dXML2ObjConverterAdapter is not None:
            initModules(sim, cc3dXML2ObjConverterAdapter)

        sim.initializeCC3D()

    print "SIMTHREAD=", simthread
    if simthread is not None:
        simthread.clearGraphicsFields()


def attachDictionaryToCells(sim):
    from CompuCell import PyAttributeAdder
    from PyDictAdder import DictAdder
    # from sys import getrefcount
    adder = PyAttributeAdder()
    # adder.registerRefChecker(getrefcount)
    dictAdder = DictAdder()
    adder.registerAdder(dictAdder)
    potts = sim.getPotts()
    potts.registerAttributeAdder(adder.getPyAttributeAdderPtr())
    return adder, dictAdder  # returning those two objects ensures that they will not be garbage collected. They are needed for proper functioning of the attribute adder


def attachListToCells(sim):
    raise AttributeError(
        'List cell attributes are no longer supported by CompuCell3D. Please use dictionaries they are much more convenient and you can always insert Python list into a dictionary.')


# # # old code
#     from CompuCell import PyAttributeAdder
#     from PyListAdder import ListAdder
#     #from sys import getrefcount
#     adder=PyAttributeAdder()
#     #adder.registerRefChecker(getrefcount)
#     listAdder=ListAdder()
#     adder.registerAdder(listAdder)
#     potts=sim.getPotts()
#     potts.registerAttributeAdder(adder.getPyAttributeAdderPtr())
#     return adder,listAdder #returning those two objects ensures that they will not be garbage collected. They are needed for proper functioning of the attribute adder

def extraInitSimulationObjects(sim, simthread, _restartEnabled=False):
    if playerType == "CMLResultReplay":
        simthread.preStartInit()
        simthread.postStartInit()
    else:

        sim.extraInit()  # after all xml steppables and plugins have been loaded we call extraInit to complete initialization

        # 62 mb    

        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        if simthread is not None and playerType != "CML":
            simthread.preStartInit()

        if not _restartEnabled:  # start function does not get called during restart
            sim.start()
        # 71 mb


        # print 'extraInitSimulationObjects 1'
        # import time
        # time.sleep(5)        


        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        # print 'extraInitSimulationObjects 2'
        # import time
        # time.sleep(5)        

        if simthread is not None and playerType != "CML":
            simthread.postStartInit()


def createVectorFieldCellLevelPy(_fieldName):
    global cmlFieldHandler  # rwh2
    global fieldRegistry
    import string
    fieldName = string.replace(_fieldName, " ", "_")  # replacing spaces with underscore

    if playerType == "CML":
        field = cmlFieldHandler.fieldStorage.createVectorFieldCellLevelPy(fieldName)
        fieldRegistry.addNewField(field, _fieldName, VECTOR_FIELD_CELL_LEVEL)
        return field
    else:
        field = simulationThreadObject.callingWidget.fieldStorage.createVectorFieldCellLevelPy(fieldName)
        fieldRegistry.addNewField(field, _fieldName, VECTOR_FIELD_CELL_LEVEL)
        return field


# # # def createVectorFieldPy(_dim,_fieldName):
# # #     global cmlFieldHandler  #rwh2
# # #     global fieldRegistry
# # #     import string
# # #     fieldName = string.replace(_fieldName," ","_") # replacing spaces with underscore

# # #     if playerType=="CML":
# # #         field = cmlFieldHandler.fieldStorage.createVectorFieldPy(_dim,fieldName)
# # #         fieldRegistry.addNewField(field,_fieldName,VECTOR_FIELD)
# # #         return field
# # #     else:
# # #         field = simulationThreadObject.callingWidget.fieldStorage.createVectorFieldPy(_dim,fieldName)
# # #         fieldRegistry.addNewField(field,_fieldName,VECTOR_FIELD)
# # #         return field


def createVectorFieldPy(_dim, _fieldName):
    global cmlFieldHandler  # rwh2
    global fieldRegistry
    import string
    fieldName = string.replace(_fieldName, " ", "_")  # replacing spaces with underscore
    import numpy as np

    if playerType == "CML":
        fieldNP = np.zeros(shape=(_dim.x, _dim.y, _dim.z, 3), dtype=np.float32)
        ndarrayAdapter = cmlFieldHandler.fieldStorage.createVectorFieldPy(_dim, _fieldName)
        ndarrayAdapter.initFromNumpy(
            fieldNP)  # initializing  numpyAdapter using numpy array (copy dims and data ptr)
        fieldRegistry.addNewField(ndarrayAdapter, _fieldName, VECTOR_FIELD)
        fieldRegistry.addNewField(fieldNP, _fieldName + '_npy', VECTOR_FIELD_NPY)
        return fieldNP

    else:

        fieldNP = np.zeros(shape=(_dim.x, _dim.y, _dim.z, 3), dtype=np.float32)
        ndarrayAdapter = simulationThreadObject.callingWidget.fieldStorage.createVectorFieldPy(_dim, _fieldName)
        ndarrayAdapter.initFromNumpy(
            fieldNP)  # initializing  numpyAdapter using numpy array (copy dims and data ptr)
        fieldRegistry.addNewField(ndarrayAdapter, _fieldName, VECTOR_FIELD)
        fieldRegistry.addNewField(fieldNP, _fieldName + '_npy', VECTOR_FIELD_NPY)
        return fieldNP


def clearVectorFieldPy(_dim, _vectorField):
    global cmlFieldHandler  # rwh2
    if playerType == "CML":
        cmlFieldHandler.fieldStorage.clearVectorFieldPy(_dim, _vectorField)
    else:
        simulationThreadObject.callingWidget.fieldStorage.clearVectorFieldPy(_dim, _vectorField)


def vtkScriptCallback(arg):
    global customVisStorage  # see above, l.140:  customVisStorage = CustomVisStorage()
    #  import sys
    #  print MYMODULENAME,'  ------------>>>>>>  vtkScriptCallback(),  arg=',arg    # --> {'my_field1': (vtkOpenGLActor)0x1261efaa0}
    #  print MYMODULENAME,'    vtkScriptCallback(),  customVisStorage =',customVisStorage
    #  print MYMODULENAME,'    vtkScriptCallback(),  customVisStorage.getVisData(arg.keys()[0]).getScript() =', \
    #           customVisStorage.getVisData(arg.keys()[0]).getScript()

    myobj = customVisStorage.getVisData(arg.keys()[0])
    #  print MYMODULENAME,'type(myobj)=',type(myobj)
    #  print MYMODULENAME,'dir(myobj)=',dir(myobj)
    actorsDict = myobj.getActorDict()
    #  print MYMODULENAME,' ------->> actorsDict=',actorsDict  # = {'my_field1': 'vtkActor', '/Users/heiland/dev/vtk-cc3d-play/runpy_cone.py': 'customVTKScript'}
    #  print MYMODULENAME,'  vtkScriptCallback(),  sys.path=',sys.path
    #  sys.argv=["",arg.keys()[0]]
    #    sys.argv=["",arg.values()[0]]
    #    sys.argv=["",str(arg.values()[0])]   # -> error
    #  print MYMODULENAME,'  vtkScriptCallback(),  sys.argv=',sys.argv    # e.g. sys.argv= ['', 'my_field1']
    #    execfile("/Users/heiland/dev/vtk-cc3d-play/ConeSimple.py")  # what about the namespace?
    #  import subprocess
    #  p = subprocess.Popen(["/Users/heiland/dev/vtk-cc3d-play/ConeSimple2.py","dummyArg"],shell=False)
    #  print p.communicate()

    #  if simulationThreadObject.sim = sim
    #  print MYMODULENAME,'  vtkScriptCallback(),  simulationThreadObject=',simulationThreadObject  # <Simulation.CMLResultReader.CMLResultReader
    #  print MYMODULENAME,'  vtkScriptCallback(),  simulationThreadObject.sim=',simulationThreadObject.sim  # None
    #  print MYMODULENAME,'  vtkScriptCallback(),  simulationThreadObject.currentFileName  ='simulationThreadObject.currentFileName  ,
    #  if simulationThreadObject:
    #      print MYMODULENAME,'  vtkScriptCallback(),  dir(simulationThreadObject)=',dir(simulationThreadObject)

    import runpy
    #  myscript = "/Users/heiland/dev/vtk-cc3d-play/bubbles1.py"
    #  myscript = "/Users/heiland/dev/vtk-cc3d-play/runpy_cone.py"
    myscript = actorsDict['customVTKScript']
    #  ns = runpy.run_path("/Users/heiland/dev/vtk-cc3d-play/runpy_cone.py", {'foo':13, 'actorObj':arg['my_field1']} )
    ns = runpy.run_path(myscript, {'actorObj': arg.values()[0],
                                   'vtkFile': simulationThreadObject.currentFileName})


#  print MYMODULENAME,'  vtkScriptCallback(),  ns["coneActor"]=',ns["coneActor"]   # LOTS of output
#  print MYMODULENAME,'  vtkScriptCallback(),  type(ns["coneActor"])=',type(ns["coneActor"])
#  print MYMODULENAME,'  vtkScriptCallback(),  type(ns["my_field1"])=',type(ns["my_field1"])

#  import vtk
#  print '\n================================================='
#  print '      hey,  I am in vtkScriptCallback'
#  print '=================================================\n'
#
#  cone = vtk.vtkConeSource()
#  cone.SetHeight( 33.0 )
#  cone.SetRadius( 10.0 )
#  coneMapper = vtk.vtkPolyDataMapper()
##  coneMapper.SetInputConnection( cone.GetOutputPort() )
#  coneMapper.SetInput( cone.GetOutput() )
#
##  coneActor = actorsDict["myActor"]
##  coneActor = customVisStorage.actorsDict["my_field1"]
##  coneActor = customVisStorage.getVisData("my_field1")

#  myobj = customVisStorage.getVisData("my_field1")
#  myobj = customVisStorage.getVisData(arg.keys()[0])
#  print MYMODULENAME,'vtkScriptCallback():  type(myobj)=',type(myobj)
#  print MYMODULENAME,'vtkScriptCallback():  dir(myobj)=',dir(myobj)
#  actorsDict = myobj.getActorDict()
##  actorsDict['my_field1'] = ns["my_field1"]
#  print MYMODULENAME,'vtkScriptCallback():  actorsDict=',actorsDict  # = {'my_field1': 'vtkActor'}
#  arg['my_field1'] = ns["my_field1"]
##  coneActor = actorsDict["my_field1"]
##  print 'type(coneActor)=',type(coneActor)  # 'str'
##  coneActor.SetMapper( coneMapper )
#  arg['my_field1'].SetMapper( coneMapper )


def createCustomVisPy(_visName):  # called from Python steppable doing custom vis; e.g. called with ("CustomCone")
    global customVisStorage  # see above, l.140:  customVisStorage = CustomVisStorage()

    customVisStorage.addNewVisData(_visName)
    return customVisStorage.getVisData(_visName)

    # customVisStorage.visDataDict[_name]=CustomVisData(_name)
    # return customVisStorage.visDataDict[_name]    


# def createCustomVisPy(_name):
# try:
# import networkx as nx
# except ImportError,e:
# return None

# global customVisStorage
# global customActorsStorage

# customVisStorage.visDataDict[_name]={}
# customVisStorage.callbackFunctionDict[_name]=None
# # customActorsStorage.actorsDict[_name]={}

# return customVisStorage.visDataDict[_name],customVisStorage.callbackFunctionDict
# # ,customActorsStorage.actorsDict


def createScalarFieldPy(_dim, _fieldName):
    return createFloatFieldPy(_dim, _fieldName)


def createFloatFieldPy(_dim, _fieldName):
    global cmlFieldHandler  # rwh2
    global fieldRegistry
    import string
    fieldName = string.replace(_fieldName, " ", "_")  # replacing spaces with underscore

    if playerType == "CML":
        import numpy as np
        fieldNP = np.zeros(shape=(_dim.x, _dim.y, _dim.z), dtype=np.float32)
        ndarrayAdapter = cmlFieldHandler.fieldStorage.createFloatFieldPy(_dim, _fieldName)
        ndarrayAdapter.initFromNumpy(fieldNP)  # initializing  numpyAdapter using numpy array (copy dims and data ptr)
        fieldRegistry.addNewField(ndarrayAdapter, _fieldName, SCALAR_FIELD)
        fieldRegistry.addNewField(fieldNP, _fieldName + '_npy', SCALAR_FIELD_NPY)
        return fieldNP
    else:
        import numpy as np
        fieldNP = np.zeros(shape=(_dim.x, _dim.y, _dim.z), dtype=np.float32)
        ndarrayAdapter = simulationThreadObject.callingWidget.fieldStorage.createFloatFieldPy(_dim, _fieldName)
        ndarrayAdapter.initFromNumpy(fieldNP)  # initializing  numpyAdapter using numpy array (copy dims and data ptr)
        fieldRegistry.addNewField(ndarrayAdapter, _fieldName, SCALAR_FIELD)
        fieldRegistry.addNewField(fieldNP, _fieldName + '_npy', SCALAR_FIELD_NPY)
        return fieldNP


def createScalarFieldCellLevelPy(_fieldName):
    global cmlFieldHandler  # rwh2
    global fieldRegistry
    import string
    fieldName = string.replace(_fieldName, " ", "_")  # replacing spaces with underscore
    if playerType == "CML":
        field = cmlFieldHandler.fieldStorage.createScalarFieldCellLevelPy(_fieldName)
        fieldRegistry.addNewField(field, _fieldName, SCALAR_FIELD_CELL_LEVEL)
        return field
    else:
        field = simulationThreadObject.callingWidget.fieldStorage.createScalarFieldCellLevelPy(_fieldName)
        fieldRegistry.addNewField(field, _fieldName, SCALAR_FIELD_CELL_LEVEL)
        return field


def doNotOutputField(_fieldName):
    global cmlFieldHandler  # rwh2
    #    print MYMODULENAME,"doNotOutputField():  cmlFieldHandler = ",cmlFieldHandler
    if cmlFieldHandler:
        cmlFieldHandler.doNotOutputField(_fieldName)
        #        print "\n\n\n\n"+ MYMODULENAME +"  cmlFieldHandler.doNotOutputFieldList=",cmlFieldHandler.doNotOutputFieldList,"\n\n\n"
        # sys.exit()


def getNameForSimDir(_simulationFileName, _preferedWorkspaceDir=""):
    # print "_simulationFileName=",_simulationFileName
    simulationFileName = str(_simulationFileName)
    fullFileName = os.path.abspath(simulationFileName)
    # filePath=os.path.dirname(fullFileName)
    (filePath, baseFileName) = os.path.split(fullFileName)

    # we store everything in the CC3D workspace directory
    if _preferedWorkspaceDir != "":
        filePath = os.path.abspath(_preferedWorkspaceDir)
    else:
        filePath = os.path.join(os.path.expanduser('~'), 'CC3DWorkspace')

    if not os.path.exists(filePath):
        os.mkdir(filePath)

    # import string

    baseFileNameForDirectory = baseFileName.replace('.', '_')
    print "baseFileName=", baseFileNameForDirectory

    import time
    timeNameExtension = "_" + time.strftime("%m", time.localtime()) + "_" + time.strftime("%d",
                                                                                          time.localtime()) + "_" + time.strftime(
        "%Y", time.localtime()) \
                        + "_" + time.strftime("%H", time.localtime()) + "_" + time.strftime("%M",
                                                                                            time.localtime()) + "_" + time.strftime(
        "%S", time.localtime())

    screenshotDirectoryName = os.path.join(filePath, baseFileNameForDirectory + timeNameExtension)
    return (screenshotDirectoryName, baseFileNameForDirectory)


def set_custom_output_directory(_dir):
    global screenshotDirectoryName
    screenshotDirectoryName = _dir

    global customScreenshotDirectoryName
    customScreenshotDirectoryName = _dir


def makeCustomSimDir(_simDir, _simulationFileName=''):
    if _simulationFileName != '':
        screenshotDirectoryName, baseFileNameForDirectory = getNameForSimDir(_simulationFileName, '')
    else:
        baseFileNameForDirectory = os.path.basename(_simDir)

    if not os.path.isdir(_simDir):

        os.mkdir(_simDir)

        return (_simDir, baseFileNameForDirectory)
    else:
        return ('', '')
        # raise OSError("Requested output directory %s already exists. Please edit python script where you set the location of the output dir"%_simDir)


def makeSimDir(_simulationFileName, _preferedWorkspaceDir=""):
    screenshotDirectoryName, baseFileNameForDirectory = getNameForSimDir(_simulationFileName, _preferedWorkspaceDir)
    if not os.path.isdir(screenshotDirectoryName):
        os.mkdir(screenshotDirectoryName)
        return (screenshotDirectoryName, baseFileNameForDirectory)
    else:
        return ("", "")


def convert_time_interval_to_hmsm(_timeInterval):
    timeInterval = int(_timeInterval)
    hours = timeInterval / (3600 * 1000)
    minutesInterval = timeInterval % (3600 * 1000)
    minutes = minutesInterval / (60 * 1000)
    secondsInterval = minutesInterval % (60 * 1000)
    seconds = secondsInterval / (1000)
    miliseconds = secondsInterval % (1000)

    out_str = ''
    if hours:
        out_str = str(hours) + " h : " + str(minutes) + " m : " + str(seconds) + " s : " + str(miliseconds) + " ms"

    elif minutes:
        out_str = str(minutes) + " m : " + str(seconds) + " s : " + str(miliseconds) + " ms"

    elif seconds:
        out_str = str(seconds) + " s : " + str(miliseconds) + " ms"

    else:
        out_str = str(miliseconds) + " ms"

    return out_str + ' = ' + str(_timeInterval / 1000.0) + ' s'


def print_profiling_report(py_steppable_profiler_report, compiled_code_run_time, total_run_time):
    myFormat = '{:>32.32}: {:11.2f} ({:5.1%})'

    print '\n\n'
    print '------------------PERFORMANCE REPORT:----------------------'
    print '-----------------------------------------------------------'
    print "TOTAL RUNTIME ", convert_time_interval_to_hmsm(total_run_time)
    print '-----------------------------------------------------------'
    print '-----------------------------------------------------------'
    print 'PYTHON STEPPABLE RUNTIMES'
    # print py_steppable_profiler_report

    totStepTime = 0

    for steppableName, steppableObjectHash, run_time_ms in py_steppable_profiler_report:
        print(myFormat.format(steppableName, int(run_time_ms) / 1000., int(run_time_ms) / total_run_time))

        totStepTime += run_time_ms

    print '-----------------------------------------------------------'

    print(myFormat.format('Total Steppable Time', int(totStepTime) / 1000., int(totStepTime) / total_run_time))

    print(myFormat.format('Compiled Code (C++) Run Time', int(compiled_code_run_time) / 1000.,
                          int(compiled_code_run_time) / total_run_time))

    print(myFormat.format('Other Time', int(total_run_time - compiled_code_run_time - totStepTime) / 1000., \
                          int(total_run_time - compiled_code_run_time - totStepTime) / total_run_time))

    print '-----------------------------------------------------------'
    print


    # print '\n\n\n'
    # print '------------------PERFORMANCE REPORT:----------------------'
    # print '-----------------------------------------------------------'
    # print "TOTAL SIMULATION RUNTIME ", convert_time_interval_to_hmsm(total_run_time)
    # print 'COMPILED CODE RUN TIME (C++):', convert_time_interval_to_hmsm(compiled_code_run_time)
    # print '-----------------------------------------------------------'
    # print 'PYTHON STEPPABLE RUNTIME:'
    # print '-----------------------------------------------------------'
    # print py_steppable_profiler_report
    # print '-----------------------------------------------------------'


def stopSimulation():
    global userStopSimulationFlag
    userStopSimulationFlag = True


def reset_current_step(step):
    global current_step
    current_step = step

    print 'AFTER RESETTING current_step=', current_step


def mainLoopNewPlayer(sim, simthread, steppableRegistry=None, _screenUpdateFrequency=None):
    global cmlFieldHandler  # rwh2
    global globalSteppableRegistry  # rwh2
    globalSteppableRegistry = steppableRegistry
    import time
    global userStopSimulationFlag
    userStopSimulationFlag = False
    t1 = time.time()
    print 'SIMULATION FILE NAME=', simthread.getSimFileName()
    global simulationFileName
    simulationFileName = simthread.getSimFileName()

    import weakref
    # restart manager
    import RestartManager
    restartManager = RestartManager.RestartManager(sim)

    simthread.restartManager = weakref.ref(restartManager)

    # restartEnabled=restartManager.restartEnabled()
    restartEnabled = restartManager.restartEnabled()
    sim.setRestartEnabled(restartEnabled)
    # restartEnabled=False
    if restartEnabled:
        print 'WILL RESTART SIMULATION'
        restartManager.loadRestartFiles()
    else:
        print 'WILL RUN SIMULATION FROM BEGINNING'

    extraInitSimulationObjects(sim, simthread, restartEnabled)

    # simthread.waitForInitCompletion()
    simthread.waitForPlayerTaskToFinish()

    runFinishFlag = True

    if not steppableRegistry is None:
        steppableRegistry.init(sim)

        if not restartEnabled:  # start function does not get called during restart
            steppableRegistry.start()
        # else:
        #     print 'RESTARTING STEERING PANEL'
        #     steppableRegistry.restart_steering_panel()

        # # restoring plots
        #             global viewManager
        #             viewManager.plotManager.restore_plots_layout()

        global customVisStorage

    simthread.steppablePostStartPrep()
    simthread.waitForPlayerTaskToFinish()

    # #restart manager
    # import RestartManager
    # restartManager=RestartManager.RestartManager(sim)
    restartManager.prepareRestarter()
    beginingStep = restartManager.getRestartStep()

    if restartEnabled:
        print 'RESTARTING STEERING PANEL'
        steppableRegistry.restart_steering_panel()

    # restartManager.setupRestartOutputDirectory()

    screenUpdateFrequency = 1

    xmlDescriptionFileWritten = False

    compiled_code_run_time = 0.0

    global current_step
    current_step = beginingStep

    # when num steps are declared at the CML
    # they have higher precedence than the number of MCS than declared in the simulation file
    global cml_args
    if cml_args and cml_args.numSteps:
        sim.setNumSteps(cml_args.numSteps)

    # for i in range(beginingStep,sim.getNumSteps()):
    while True:

        if simthread is not None:

            simthread.beforeStep(current_step)
            # will wait until initialization of the player is finished before proceeding further
            if cmlFieldHandler and not xmlDescriptionFileWritten:  # rwh - global defn of cmlFieldHandler ??
                #                print MYMODULENAME,"mainLoopNewPlayer(),  BEFORE getInfoAboutFields"
                #                print MYMODULENAME,"mainLoopNewPlayer(), cmlFieldHandler.sim=",cmlFieldHandler.sim
                cmlFieldHandler.getInfoAboutFields()
                cmlFieldHandler.writeXMLDescriptionFile()
                xmlDescriptionFileWritten = True
            # print MYMODULENAME,"cmlFieldHandler.doNotOutputFieldList=",cmlFieldHandler.doNotOutputFieldList

            if simthread.getStopSimulation() or userStopSimulationFlag:
                runFinishFlag = False;
                break
                # calling Python steppables which are suppose to run before MCS - e.g. secretion steppable
        if not steppableRegistry is None:
            steppableRegistry.stepRunBeforeMCSSteppables(current_step)

        restartManager.outputRestartFiles(current_step)

        compiled_code_begin = time.time()

        sim.step(current_step)  # steering using steppables

        compiled_code_end = time.time()

        compiled_code_run_time += (compiled_code_end - compiled_code_begin) * 1000

        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        simthread.steerUsingGUI(sim)  # steering using GUI. GUI steering overrides steering done in the steppables

        if not steppableRegistry is None:
            steppableRegistry.step(current_step)
        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()
        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        screenUpdateFrequency = simthread.getScreenUpdateFrequency()
        #        imgOutFlag = simthread.getImageOutputFlag()
        imgOutFlag = False
        screenshotFrequency = simthread.getScreenshotFrequency()
        latticeFlag = simthread.getLatticeOutputFlag()
        latticeFrequency = simthread.getLatticeOutputFrequency()

        if simthread is not None:
            if (current_step % screenUpdateFrequency == 0) or (
                        imgOutFlag and (current_step % screenshotFrequency == 0)):
                simthread.loopWork(current_step)
                simthread.loopWorkPostEvent(current_step)
                screenUpdateFrequency = simthread.getScreenUpdateFrequency()

        current_step += 1

        if current_step >= sim.getNumSteps():
            break

    print "END OF SIMULATION  "

    if runFinishFlag:
        # we emit request to finish simulation
        simthread.emitFinishRequest()
        # then we wait for GUI thread to unlock the finishMutex - it will only happen when all tasks in the GUI thread are completed (especially those that need simulator object to stay alive)
        simthread.finishMutex.lock()
        simthread.finishMutex.unlock()
        # at this point GUI thread finished all the tasks for which simulator had to stay alive  and we can proceed to destroy simulator

        sim.finish()
        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())
        steppableRegistry.finish()
        sim.cleanAfterSimulation()
        simthread.simulationFinishedPostEvent(True)
        print "CALLING FINISH"
    else:
        sim.cleanAfterSimulation()
        # sim.unloadModules()
        print "CALLING UNLOAD MODULES NEW PLAYER"
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            simthread.simulationFinishedPostEvent(True)

    t2 = time.time()

    print_profiling_report(py_steppable_profiler_report=steppableRegistry.get_profiler_report(),
                           compiled_code_run_time=compiled_code_run_time, total_run_time=(t2 - t1) * 1000.0)

    # In exception handlers you have to call sim.finish to unload the plugins .
    # We may need to introduce new funuction name (e.g. unload) because finish does more than unloading

def mainLoopCLI(sim, simthread, steppableRegistry=None, _screenUpdateFrequency=None):
    print '\n### Staring main Loop CLI'
    global cmlFieldHandler  # rwh2
    global globalSteppableRegistry  # rwh2
    globalSteppableRegistry = steppableRegistry
    import ProjectFileStore

    extraInitSimulationObjects(sim, simthread)

    global customScreenshotDirectoryName
    global cc3dSimulationDataHandler
    global simulationPaths
    if customScreenshotDirectoryName:
        makeCustomSimDir(customScreenshotDirectoryName)
        if cc3dSimulationDataHandler is not None:
            cc3dSimulationDataHandler.copySimulationDataFiles(customScreenshotDirectoryName)
            simulationPaths.setSimulationResultStorageDirectoryDirect(customScreenshotDirectoryName)

    runFinishFlag = True

    if not steppableRegistry is None:
        steppableRegistry.init(sim)
        steppableRegistry.start()
    # init fieldWriter
    if cmlFieldHandler:
        cmlFieldHandler.fieldWriter.init(sim)
        cmlFieldHandler.getInfoAboutFields()
        cmlFieldHandler.outputFrequency = ProjectFileStore.outputFrequency
        cmlFieldHandler.outputFileCoreName = "output"

        cmlFieldHandler.prepareSimulationStorageDir(
            os.path.join(ProjectFileStore.outputDirectoryPath, "LatticeData"))
        cmlFieldHandler.setMaxNumberOfSteps(
            sim.getNumSteps())  # will determine the length text field  of the step number suffix
        cmlFieldHandler.writeXMLDescriptionFile()  # initialization of the cmlFieldHandler is done - we can write XML description file

        # self.simulationXMLFileName=""
        # self.simulationPythonScriptName=""

        print "simulationPaths XML=", simulationPaths.simulationXMLFileName
        print "simulationPaths PYTHON=", simulationPaths.simulationPythonScriptName


    global current_step
    current_step = 0

    # when num steps are declared at the CML
    # they have higher precedence than the number of MCS than declared in the simulation file

    # for current_step in range(sim.getNumSteps()):
    while True:
        # calling Python steppables which are suppose to run before MCS - e.g. secretion steppable
        if userStopSimulationFlag:
            runFinishFlag = False;
            break

        if not steppableRegistry is None:
            steppableRegistry.stepRunBeforeMCSSteppables(current_step)


        sim.step(current_step)  # steering using steppables


        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        if not steppableRegistry is None:
            steppableRegistry.step(current_step)

        if cmlFieldHandler.outputFrequency and not (current_step % cmlFieldHandler.outputFrequency):
            #            print MYMODULENAME,' mainLoopCML: cmlFieldHandler.writeFields(i), i=',i
            cmlFieldHandler.writeFields(current_step)
            # cmlFieldHandler.fieldWriter.addCellFieldForOutput()
            # cmlFieldHandler.fieldWriter.writeFields(cmlFieldHandler.outputFileCoreName+str(i)+".vtk")
            # cmlFieldHandler.fieldWriter.clear()

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()
        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        current_step += 1
        # print 'sim.getNumSteps()=',sim.getNumSteps()
        if current_step >= sim.getNumSteps():
            break

    print "END OF SIMULATION  "
    if runFinishFlag:
        sim.finish()
        steppableRegistry.finish()
        sim.cleanAfterSimulation()
    else:
        sim.cleanAfterSimulation()
        print "CALLING UNLOAD MODULES"


    # In exception handlers you have to call sim.finish to unload the plugins .
    # We may need to introduce new funuction name (e.g. unload) because finish does more than unloading

def mainLoopCML(sim, simthread, steppableRegistry=None, _screenUpdateFrequency=None):
    global cmlFieldHandler  # rwh2
    global globalSteppableRegistry  # rwh2
    globalSteppableRegistry = steppableRegistry

    import time
    t1 = time.time()
    # print 'SIMULATION FILE NAME=',simthread.getSimFileName()
    # global simulationFileName
    # simulationFileName=simthread.getSimFileName()
    # restart manager
    import RestartManager
    restartManager = RestartManager.RestartManager(sim)
    simthread.restartManager = restartManager

    # restartEnabled=restartManager.restartEnabled()
    restartEnabled = restartManager.restartEnabled()
    sim.setRestartEnabled(restartEnabled)
    # restartEnabled=False
    if restartEnabled:
        print 'WILL RESTART SIMULATION'
        restartManager.loadRestartFiles()
    else:
        print 'WILL RUN SIMULATION FROM BEGINNING'

    extraInitSimulationObjects(sim, simthread)

    global customScreenshotDirectoryName
    global cc3dSimulationDataHandler
    global simulationPaths
    if customScreenshotDirectoryName:
        makeCustomSimDir(customScreenshotDirectoryName)
        if cc3dSimulationDataHandler is not None:
            cc3dSimulationDataHandler.copySimulationDataFiles(customScreenshotDirectoryName)
            simulationPaths.setSimulationResultStorageDirectoryDirect(customScreenshotDirectoryName)

    runFinishFlag = True

    if not steppableRegistry is None:
        steppableRegistry.init(sim)
        steppableRegistry.start()
    # init fieldWriter    
    if cmlFieldHandler:
        cmlFieldHandler.fieldWriter.init(sim)
        cmlFieldHandler.getInfoAboutFields()
        cmlFieldHandler.outputFrequency = cmlParser.outputFrequency
        cmlFieldHandler.outputFileCoreName = cmlParser.outputFileCoreName

        cmlFieldHandler.prepareSimulationStorageDir(
            os.path.join(simulationPaths.simulationResultStorageDirectory, "LatticeData"))
        cmlFieldHandler.setMaxNumberOfSteps(
            sim.getNumSteps())  # will determine the length text field  of the step number suffix
        cmlFieldHandler.writeXMLDescriptionFile()  # initialization of the cmlFieldHandler is done - we can write XML description file

        # self.simulationXMLFileName=""
        # self.simulationPythonScriptName=""

        print "simulationPaths XML=", simulationPaths.simulationXMLFileName
        print "simulationPaths PYTHON=", simulationPaths.simulationPythonScriptName

    restartManager.prepareRestarter()
    beginingStep = restartManager.getRestartStep()

    compiled_code_run_time = 0.0

    global current_step
    current_step = beginingStep

    # when num steps are declared at the CML
    # they have higher precedence than the number of MCS than declared in the simulation file
    global cml_args
    if cml_args and cml_args.numSteps:
        sim.setNumSteps(cml_args.numSteps)

    # for current_step in range(sim.getNumSteps()):
    while True:
        # calling Python steppables which are suppose to run before MCS - e.g. secretion steppable
        if userStopSimulationFlag:
            runFinishFlag = False;
            break

        if not steppableRegistry is None:
            steppableRegistry.stepRunBeforeMCSSteppables(current_step)

        restartManager.outputRestartFiles(current_step)

        compiled_code_begin = time.time()

        sim.step(current_step)  # steering using steppables

        compiled_code_end = time.time()

        compiled_code_run_time += (compiled_code_end - compiled_code_begin) * 1000

        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        if not steppableRegistry is None:
            steppableRegistry.step(current_step)

        if cmlFieldHandler.outputFrequency and not (current_step % cmlFieldHandler.outputFrequency):
            #            print MYMODULENAME,' mainLoopCML: cmlFieldHandler.writeFields(i), i=',i
            cmlFieldHandler.writeFields(current_step)
            # cmlFieldHandler.fieldWriter.addCellFieldForOutput()
            # cmlFieldHandler.fieldWriter.writeFields(cmlFieldHandler.outputFileCoreName+str(i)+".vtk")
            # cmlFieldHandler.fieldWriter.clear()

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()
        if sim.getRecentErrorMessage() != "":
            raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

        current_step += 1
        # print 'sim.getNumSteps()=',sim.getNumSteps()
        if current_step >= sim.getNumSteps():
            break

    print "END OF SIMULATION  "
    if runFinishFlag:
        sim.finish()
        steppableRegistry.finish()
        sim.cleanAfterSimulation()
    else:
        sim.cleanAfterSimulation()
        print "CALLING UNLOAD MODULES"

    t2 = time.time()

    broadcast_simulation_return_value()
    # raise RuntimeError('after broadcasting return value')
    print_profiling_report(py_steppable_profiler_report=steppableRegistry.get_profiler_report(),
                           compiled_code_run_time=compiled_code_run_time, total_run_time=(t2 - t1) * 1000.0)

    # In exception handlers you have to call sim.finish to unload the plugins .
    # We may need to introduce new funuction name (e.g. unload) because finish does more than unloading


def set_return_value_tag(tag='generic_label'):
    """
    Sets return value tag
    :param tag:  arbitrary string allowing easier identification of the results
    :return: None
    """
    print 'setting return_value_tag=',tag

    global simulation_return_value_tag
    simulation_return_value_tag = tag

def set_simulation_return_value(val, tag=None):
    """
    records simulation return value - typically to be sent back to the server running some type of optimization algorithm
    :param val : simulation return value
    :param tag:  arbitrary string allowing easier identification of the results
    :return: None
    """
    global simulation_return_value
    global simulation_return_value_tag

    simulation_return_value = val

    if tag is not None:
        simulation_return_value_tag = tag


def get_simulation_return_value():
    """
    Returns simulation return value and siulation return value tag
    :return: tuple ( simulation_return_value, simulation_return_value_tag)
    """
    global simulation_return_value
    global simulation_return_value_tag

    return simulation_return_value, simulation_return_value_tag


def set_push_address(addr):
    """
    sets push address to be used with ZMQ PUSH/PUL protocol
    :param addr {str}: push address
    :return:  None
    """
    global push_address
    push_address = addr

def broadcast_simulation_return_value():
    """
    Sends simulation return value over zmq channel with address given by
    global push_address. In case push_address is None or simulation return value has not been set
    this function returns early
    :return: None
    """
    print 'INSIDE broadcast_simulation_return_value'

    global push_address

    print 'push_address=', push_address

    simulation_return_value, simulation_return_value_tag = get_simulation_return_value()

    print 'simulation_return_value=', simulation_return_value
    if simulation_return_value is None:
        return

    if push_address is None:
        return

    try:
        import zmq
    except ImportError:
        print 'PLEASE INSTALL PYZMQ BEFORE ATTAMPTING TO USE broadcast_simulation_return_value function'
        return

    context = zmq.Context()

    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect(push_address)

    # result = {'tag': tag, 'return_value': simulation_return_value}
    result = {'return_value_tag': simulation_return_value_tag, 'return_value': simulation_return_value,'abort':False}
    print 'will send the result ', result
    consumer_sender.send_json(result)
    print 'result sent'


def mainLoop(sim, simthread, steppableRegistry=None, _screenUpdateFrequency=None):
    global playerType
    #    print MYMODULENAME,"playerType=",playerType
    #    import pdb; pdb.set_trace()

    if invokeMethod == "CLI":
        return mainLoopCLI(sim, simthread, steppableRegistry, _screenUpdateFrequency)
    if playerType == "CML":
        return mainLoopCML(sim, simthread, steppableRegistry, _screenUpdateFrequency)
    if playerType == "CMLResultReplay":
        return mainLoopCMLReplay(sim, simthread, steppableRegistry, _screenUpdateFrequency)

    # print MYMODULENAME,' mainLoop:  simulationFileName =',simulationFileName
    if simulationThreadObject is None:
        print MYMODULENAME, ' mainLoop:  error, simulationThreadObject is None (no longer an OldPlayer)'
    # return mainLoopOldPlayer(sim, simthread, steppableRegistry, _screenUpdateFrequency )
    else:
        return mainLoopNewPlayer(sim, simthread, steppableRegistry, _screenUpdateFrequency)

