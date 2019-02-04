# -*- coding: utf-8 -*-
from PyQt5.QtCore import *
import CompuCell
import PlayerPython
from cc3d.core import XMLUtils
import re
from cc3d.core.XMLUtils import CC3DXMLListPy
import CC3DXML
import cc3d.CompuCellSetup as CompuCellSetup
import vtk
import os
import os.path
from . import SimulationThread

(STOP_STATE, RUN_STATE, STEP_STATE, PAUSE_STATE) = list(range(0, 4))

MODULENAME = '---- CMLResultReader.py: '


class DataReader(QThread):
    data_read = pyqtSignal(int, name='data_read')

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.cmlReader = parent
        self.i = 0

    def setStep(self, _step):
        self.i = _step

    def run(self):

        readState = self.cmlReader.readSimulationDataNonBlocking(self.i)
        if readState:
            self.data_read.emit(self.i)
        else:
            self.data_read.emit(-1)


class CMLResultReader(SimulationThread.SimulationThread):
    data_read = pyqtSignal(int, name='data_read')

    initial_data_read = pyqtSignal(bool, name='initial_data_read')
    subsequent_data_read = pyqtSignal(int, name='subsequent_data_read')
    final_data_read = pyqtSignal(bool, name='final_data_read')

    # CONSTRUCTOR 
    def __init__(self, parent=None):
        SimulationThread.SimulationThread.__init__(self, parent)

        # NOTE: to implement synchronization between threads we use semaphores. If yuou use mutexes for this then if you lokc mutex in one thread and try to unlock
        # from another thread than on Linux it will not work. Semaphores are better for this

        # data extracted from LDS file *.dml
        self.fieldDim = None
        self.fieldDimPrevious = None
        self.ldsCoreFileName = ""
        self.currentFileName = ""
        self.ldsDir = None
        self.ldsFileList = []  # list of *.vtk files containing graphics lattice data
        self.fieldsUsed = {}
        self.sim = None
        self.simulationData = None
        self.simulationDataReader = None
        self.frequency = 0;
        self.currentStep = 0;
        self.latticeType = "Square"
        self.numberOfSteps = 0;
        self.__directAccessFlag = False
        self.__mcsDirectAccess = 0;
        self.__fileNumber = -1  # self.counter=0
        self.newFileBeingLoaded = False
        self.readFileSem = QSemaphore(1)
        self.typeIdTypeNameDict = {}
        self.__fileWriter = None
        self.typeIdTypeNameCppMap = None  # C++ map<int,string> providing mapping from type id to type name
        self.customVis = None
        self.__ui = parent
        self.__initialized = False

        self.stepCounter = 0
        self.reading = False
        self.state = STOP_STATE

        self.stay_in_current_step = False

        self.recently_read_file_number = -1

    # Python 2.6 requires importing of Example, CompuCell and Player Python modules from an instance  of QThread class (here Simulation Thread inherits from QThread)
    # Simulation Thread communicates with SimpleTabView using SignalSlot method. If we dont import the three modules then when SimulationThread emits siglan and SimpleTabView
    # processes this signal in a slot (e.g. initializeSimulationViewWidget) than calling a member function of an object from e.g. Player Python (self.fieldStorage.allocateCellField(self.fieldDim))
    # results in segfault. Python 2.5 does not have this issue. Anyway this seems to work on Linux with Python 2.6
    # This might be a problem only on Linux
    # import Example
    # import CompuCell
    # import PlayerPython

    def set_stay_in_current_step(self, flag):
        self.stay_in_current_step = flag

    def get_stay_in_current_step(self):
        return self.stay_in_current_step

    def gotData(self, _i):
        # print '\n\n\n       GOT DATA FOR STEP ',_i
        # print '\n\n'
        self.recently_read_file_number = _i

        self.dataReader.data_read.disconnect(self.gotData)
        self.reading = False

        if _i == 0 and not self.__initialized:
            self.initial_data_read.emit(True)
            self.__initialized = True
        if _i < 0:  # meaning read did not succeed
            self.setStopState()
            self.final_data_read.emit(True)
        else:
            self.subsequent_data_read.emit(_i)

    def setRunState(self):
        self.state = RUN_STATE

    def setStepState(self):
        self.state = STEP_STATE

    def setPauseState(self):
        self.state = PAUSE_STATE

    def setStopState(self):
        self.state = STOP_STATE

    def keepGoing(self):

        if self.state == RUN_STATE:
            self.step()

    def step(self):
        # ignoring step requests while reading operation is pending
        # print 'INSIDE STEP self.reading=',self.reading
        if self.reading:
            return

        # this section repeats the current step - pretends that the file was read again
        # used to give users a chance to change initial view in the graphics widget to ensure that all screenshots
        # are saved including screenshot for the first file

        if self.stay_in_current_step:
            self.stay_in_current_step = False
            self.reading = False
            if self.recently_read_file_number >= 0:
                self.subsequent_data_read.emit(self.recently_read_file_number)

            return

        print('self.stepCounter=', self.stepCounter)
        print('DIRECT ACCESS =', self.getCurrentStepDirectAccess())

        if self.__directAccessFlag:
            self.stepCounter = self.__mcsDirectAccess
            self.__directAccessFlag = False

        self.dataReader = DataReader(parent=self)
        self.dataReader.setStep(self.stepCounter)
        self.dataReader.data_read.connect(self.gotData)

        self.stepCounter += 1

        self.reading = True  # flag indicating that reading is taking place
        self.dataReader.start()

    def dimensionChange(self):
        # print 'self.fieldDimPrevious=',self.fieldDimPrevious
        # print 'self.fieldDim=',self.fieldDim
        if self.fieldDimPrevious:
            if self.fieldDimPrevious.x != self.fieldDim.x or self.fieldDimPrevious.y != self.fieldDim.y or self.fieldDimPrevious.z != self.fieldDim.z:
                return True
            else:
                return False
        else:
            return False

    def resetDimensionChangeMonitoring(self):
        # this reset is necessary to avoid recursive calls in the SimpleTabView -- this code needs to be changed because at this point it looks horrible
        self.fieldDimPrevious = self.fieldDim

    def readSimulationDataNonBlocking(self, _i):
        self.newFileBeingLoaded = True  # this flag is used to prevent calling  draw function when new data is read from hard drive
        # print "LOCKED self.newFileBeingLoaded=",self.newFileBeingLoaded
        # self.drawMutex.lock()
        if _i >= len(self.ldsFileList):
            return False

        fileName = self.ldsFileList[_i]

        self.simulationDataReader = vtk.vtkStructuredPointsReader()

        self.currentFileName = os.path.join(self.ldsDir, fileName)
        print('self.currentFileName=', self.currentFileName)

        extractedMCS = self.extractMCSNumberFromFileName(self.currentFileName)
        if extractedMCS < 0:
            extractedMCS = _i

        self.simulationDataReader.SetFileName(self.currentFileName)
        # print "path= ", os.path.join(self.ldsDir,fileName)

        dataReaderIntAddr = self.__ui.extractAddressIntFromVtkObject(self.simulationDataReader)

        # swig wrapper  on top of     vtkStructuredPointsReader.Update()  - releases GIL,
        # hence can be used in multithreaded program that does not block GUI
        self.__ui.fieldExtractor.readVtkStructuredPointsData(dataReaderIntAddr)

        # # # self.simulationDataReader.Update() # does not erlease GIL - VTK ISSUE -
        # cannot be used in multithreaded program - blocks GUI
        self.simulationData = self.simulationDataReader.GetOutput()


        self.fieldDimPrevious = self.fieldDim
        # self.fieldDimPrevious=CompuCell.Dim3D()

        dimFromVTK = self.simulationData.GetDimensions()

        self.fieldDim = CompuCell.Dim3D(dimFromVTK[0], dimFromVTK[1], dimFromVTK[2])

        self.currentStep = extractedMCS
        # # # self.currentStep = self.frequency * _i # this is how we set CMS for CML reading before
        self.setCurrentStep(self.currentStep)

        return True

    def extractMCSNumberFromFileName(self, _fileName):

        coreName, ext = os.path.splitext(os.path.basename(_fileName))


        mcs_extractor_regex = re.compile('([\D]*)([0-9]*)')
        match = re.match(mcs_extractor_regex, coreName)

        if match:
            matchGroups = match.groups()

            if matchGroups[1] != '':
                mcs_str = matchGroups[1]

                return int(mcs_str)

        return -1

    def readSimulationData(self, _i):

        self.drawMutex.lock()
        self.readFileSem.acquire()

        # this flag is used to prevent calling  draw function when new data is read from hard drive
        self.newFileBeingLoaded = True

        if _i >= len(self.ldsFileList):
            return

        fileName = self.ldsFileList[_i]

        self.simulationDataReader = vtk.vtkStructuredPointsReader()
        self.currentFileName = os.path.join(self.ldsDir, fileName)
        self.simulationDataReader.SetFileName(self.currentFileName)

        self.simulationDataReader.Update()
        self.simulationData = self.simulationDataReader.GetOutput()

        # updating fieldDim each time we read data
        self.fieldDimPrevious = self.fieldDim

        dimFromVTK = self.simulationData.GetDimensions()

        self.fieldDim = CompuCell.Dim3D(dimFromVTK[0], dimFromVTK[1], dimFromVTK[2])

        self.currentStep = self.frequency * _i
        self.setCurrentStep(self.currentStep)

        self.drawMutex.unlock()
        self.readFileSem.release()

    def extractLatticeDescriptionInfo(self, _fileName : str):
        """

        :param _fileName:
        :return:
        """
        ldsFile = os.path.abspath(_fileName)
        self.ldsDir = os.path.dirname(ldsFile)

        xml2ObjConverter = XMLUtils.Xml2Obj()
        root_element = xml2ObjConverter.Parse(ldsFile)
        dimElement = root_element.getFirstElement("Dimensions")
        self.fieldDim = CompuCell.Dim3D()
        self.fieldDim.x = int(dimElement.getAttribute("x"))
        self.fieldDim.y = int(dimElement.getAttribute("y"))
        self.fieldDim.z = int(dimElement.getAttribute("z"))
        outputElement = root_element.getFirstElement("Output")
        self.ldsCoreFileName = outputElement.getAttribute("CoreFileName")
        self.frequency = int(outputElement.getAttribute("Frequency"))
        self.numberOfSteps = int(outputElement.getAttribute("NumberOfSteps"))

        # obtaining list of files in the ldsDir
        latticeElement = root_element.getFirstElement("Lattice")
        self.latticeType = latticeElement.getAttribute("Type")

        # getting information about cell type names and cell ids. It is necessary during generation of the PIF files from VTK output
        cellTypesElements = root_element.getElements("CellType")
        listCellTypeElements = CC3DXMLListPy(cellTypesElements)
        for cellTypeElement in listCellTypeElements:
            typeName = cellTypeElement.getAttribute("TypeName")
            typeId = cellTypeElement.getAttributeAsInt("TypeId")
            self.typeIdTypeNameDict[typeId] = typeName

        # now will convert python dictionary into C++ map<int, string>     

        self.typeIdTypeNameCppMap = CC3DXML.MapIntStr()
        for typeId in list(self.typeIdTypeNameDict.keys()):
            self.typeIdTypeNameCppMap[int(typeId)] = self.typeIdTypeNameDict[typeId]


        ldsFileList = os.listdir(self.ldsDir)

        for fName in ldsFileList:
            if re.match(".*\.vtk$", fName):
                self.ldsFileList.append(fName)

        self.ldsFileList.sort()

        # extracting information about fields in the lds file
        fieldsElement = root_element.getFirstElement("Fields")
        if fieldsElement:
            fieldList = XMLUtils.CC3DXMLListPy(fieldsElement.getElements("Field"))

            for fieldElem in fieldList:

                fieldName = fieldElem.getAttribute("Name")
                self.fieldsUsed[fieldElem.getAttribute("Name")] = fieldElem.getAttribute("Type")
                if fieldElem.findAttribute("Script"):  # True or False if present
                    # ToDo:  if a "CustomVis" Type was provided, require that a "Script" was also provided; else warn user
                    customVisScript = fieldElem.getAttribute("Script")

                    self.customVis = CompuCellSetup.createCustomVisPy(fieldName)
                    self.customVis.registerVisCallbackFunction(CompuCellSetup.vtkScriptCallback)

                    self.customVis.addActor(fieldName, "vtkActor")
                    # we'll piggyback off the actorsDict
                    self.customVis.addActor("customVTKScript", customVisScript)

    def generatePIFFromVTK(self, _vtkFileName : str , _pifFileName: str) -> None:
        """
        generates PIFF from VTK file
        :param _vtkFileName:
        :param _pifFileName:
        :return:
        """
        if self.__fileWriter is None:

            self.__fileWriter = PlayerPython.FieldWriter()

        self.__fileWriter.generatePIFFileFromVTKOutput(_vtkFileName, _pifFileName, self.fieldDim.x, self.fieldDim.y,
                                                       self.fieldDim.z, self.typeIdTypeNameCppMap)

    def setCurrentStepDirectAccess(self, _mcs):
        self.__mcsDirectAccess = _mcs
        self.__directAccessFlag = True

    def getCurrentStepDirectAccess(self):
        return (self.__mcsDirectAccess, self.__directAccessFlag)

    def resetDirectAccessFlag(self):
        self.__directAccessFlag = False

    def steerUsingGUI(self, _sim):
        pass

    def run(self):
        return
        globalDict = {'simTabView': 20}
        localDict = {}
        print('self.pythonFileName=', self.pythonFileName)
        self.runUserPythonScript(self.pythonFileName, globalDict, localDict)
        return
