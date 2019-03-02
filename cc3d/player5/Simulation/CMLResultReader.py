# -*- coding: utf-8 -*-
from cc3d.core.enums import *
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
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object


class DataReader(QThread):
    data_read = pyqtSignal(int, name='data_read')

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.cml_reader = parent
        self.step = 0

    def set_step(self, step):
        self.step = step

    def run(self):
        """
        reads simulation data. emits a signal after the read finished
        :return:
        """
        read_state = self.cml_reader.read_simulation_data_non_blocking(self.step)
        if read_state:
            self.data_read.emit(self.step)
        else:
            self.data_read.emit(-1)


class CMLResultReader(SimulationThread.SimulationThread):

    data_read = pyqtSignal(int, name='data_read')
    initial_data_read = pyqtSignal(bool, name='initial_data_read')
    subsequent_data_read = pyqtSignal(int, name='subsequent_data_read')
    final_data_read = pyqtSignal(bool, name='final_data_read')

    def __init__(self, parent=None):
        SimulationThread.SimulationThread.__init__(self, parent)

        # NOTE: to implement synchronization between threads we use semaphores.
        # If you use mutexes for this then if you lock mutex in one thread and try to unlock
        # from another thread than on Linux it will not work. Semaphores are better for this

        # data extracted from LDS file *.dml
        self.fieldDim = None
        self.fieldDimPrevious = None
        self.ldsCoreFileName = ""
        self.currentFileName = ""
        self.ldsDir = None

        # list of *.vtk files containing graphics lattice data
        self.ldsFileList = []
        self.fieldsUsed = {}
        self.sim = None
        self.simulationData = None
        self.simulationDataReader = None
        self.frequency = 0
        self.currentStep = 0
        self.latticeType = "Square"
        self.numberOfSteps = 0
        self.__direct_access_flag = False
        self.__mcsDirectAccess = 0
        self.__fileNumber = -1  # self.counter=0
        self.newFileBeingLoaded = False
        self.readFileSem = QSemaphore(1)
        self.typeIdTypeNameDict = {}
        self.__fileWriter = None

        # C++ map<int,string> providing mapping from type id to type name
        self.typeIdTypeNameCppMap = None
        self.customVis = None
        self.__ui = parent
        self.__initialized = False

        self.stepCounter = 0
        self.reading = False
        self.state = STOP_STATE

        self.stay_in_current_step = False

        self.recently_read_file_number = -1

    def set_stay_in_current_step(self, flag):
        self.stay_in_current_step = flag

    def get_stay_in_current_step(self):
        return self.stay_in_current_step

    def got_data(self, file_number):
        """
        slot handling data read
        :param file_number:
        :return:
        """
        self.recently_read_file_number = file_number

        self.data_reader.data_read.disconnect(self.got_data)
        self.reading = False

        if file_number == 0 and not self.__initialized:
            self.initial_data_read.emit(True)
            self.__initialized = True


        if file_number < 0:
            # read did not succeed
            self.set_run_state(state=STOP_STATE)
            self.final_data_read.emit(True)
        else:
            # read successful
            self.subsequent_data_read.emit(file_number)

    def set_run_state(self, state:int)->None:
        """
        sets current run state
        :param state:
        :return:
        """

        self.state = state


    def keep_going(self):
        """
        executes step fcn if self.state == RUN_STATE
        :return:
        """

        if self.state == RUN_STATE:
            self.step()

    def step(self) -> None:
        """
        executes single step for CMLResultReplay
        :return:
        """

        # ignoring step requests while reading operation is pending
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

        if self.__direct_access_flag:
            self.stepCounter = self.__mcsDirectAccess
            self.__direct_access_flag = False

        self.data_reader = DataReader(parent=self)
        self.data_reader.set_step(self.stepCounter)
        self.data_reader.data_read.connect(self.got_data)

        self.stepCounter += 1

        # flag indicating that reading is taking place
        self.reading = True
        self.data_reader.start()

    def dimensionChange(self) -> bool:
        """
        event handler that processes change of dimension of the simulation
        :return: {bool} flag indicating if essentiall handler code was actually or not
        """
        if self.fieldDimPrevious:
            if self.fieldDimPrevious.x != self.fieldDim.x \
                    or self.fieldDimPrevious.y != self.fieldDim.y or self.fieldDimPrevious.z != self.fieldDim.z:
                return True
            else:
                return False
        else:
            return False

    def resetDimensionChangeMonitoring(self):
        """

        :return:
        """

        # this reset is necessary to avoid recursive calls in the
        # SimpleTabView -- this code needs to be changed because at this point it looks horrible
        self.fieldDimPrevious = self.fieldDim

    @property
    def data_ready(self):
        return self.newFileBeingLoaded

    def read_simulation_data_non_blocking(self, file_number:int)->bool:
        """
        reads content of the serialized file
        :param file_number: {int}
        :return: {bool} flag whether read was successful or not
        """

        # this flag is used to prevent calling  draw function
        self.newFileBeingLoaded = True

        # when new data is read from hard drive
        if file_number >= len(self.ldsFileList):
            return False

        fileName = self.ldsFileList[file_number]

        self.simulationDataReader = vtk.vtkStructuredPointsReader()

        self.currentFileName = os.path.join(self.ldsDir, fileName)
        print('self.currentFileName=', self.currentFileName)

        extracted_mcs = self.extract_mcs_number_from_file_name(file_name=self.currentFileName)
        if extracted_mcs < 0:
            extracted_mcs = file_number

        self.simulationDataReader.SetFileName(self.currentFileName)
        # print "path= ", os.path.join(self.ldsDir,fileName)

        data_reader_int_addr = extract_address_int_from_vtk_object(vtkObj=self.simulationDataReader)

        # swig wrapper  on top of     vtkStructuredPointsReader.Update()  - releases GIL,
        # hence can be used in multithreaded program that does not block GUI
        self.__ui.fieldExtractor.readVtkStructuredPointsData(data_reader_int_addr)

        # # # self.simulationDataReader.Update() # does not erlease GIL - VTK ISSUE -
        # cannot be used in multithreaded program - blocks GUI
        self.simulationData = self.simulationDataReader.GetOutput()

        self.fieldDimPrevious = self.fieldDim
        # self.fieldDimPrevious=CompuCell.Dim3D()

        dim_from_vtk = self.simulationData.GetDimensions()

        self.fieldDim = CompuCell.Dim3D(dim_from_vtk[0], dim_from_vtk[1], dim_from_vtk[2])

        self.currentStep = extracted_mcs
        # # # self.currentStep = self.frequency * _i # this is how we set CMS for CML reading before
        self.setCurrentStep(self.currentStep)

        self.newFileBeingLoaded = False

        return True

    def extract_mcs_number_from_file_name(self, file_name: str) -> int:
        """
        Extracts mcs from serialized file name
        :param file_name:
        :return:
        """

        core_name, ext = os.path.splitext(os.path.basename(file_name))

        mcs_extractor_regex = re.compile('([\D]*)([0-9]*)')
        match = re.match(mcs_extractor_regex, core_name)

        if match:
            match_groups = match.groups()

            if match_groups[1] != '':
                mcs_str = match_groups[1]

                return int(mcs_str)

        return -1


    def extractLatticeDescriptionInfo(self, _fileName: str):
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

    def generatePIFFromVTK(self, _vtkFileName: str, _pifFileName: str) -> None:
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
        self.__direct_access_flag = True

    def getCurrentStepDirectAccess(self):
        return (self.__mcsDirectAccess, self.__direct_access_flag)

    def resetDirectAccessFlag(self):
        self.__direct_access_flag = False

    def steerUsingGUI(self, _sim):
        """
        dummy overwrite of base class method
        :param _sim:
        :return:
        """

    def run(self):
        """
        dummy overwrite of base class method
        :return:
        """
