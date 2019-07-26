# -*- coding: utf-8 -*-
"""
This module handles  reading of serialized simulation output. It has
API of Simulation Thread and from the point of view of player
it behaves as a "regular" simulation with one exception that
instead running actual simulation during call to the step function it reads previously
saves simulation snapshot (currently stored as vtk file)
"""

from cc3d.core.enums import *
from PyQt5.QtCore import *
from cc3d.cpp import CompuCell
from cc3d.cpp import PlayerPython
from cc3d.core import XMLUtils
import re
from cc3d.core.XMLUtils import CC3DXMLListPy
from cc3d.cpp import CC3DXML
import cc3d.CompuCellSetup as CompuCellSetup
import vtk
import os
import os.path
from . import SimulationThread
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object
# from cc3d.player5.Simulation.CMLResultReader import CMLResultReader
import cc3d.player5.Simulation


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
        self.__mcs_direct_access = 0
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

    def set_stay_in_current_step(self, flag: bool) -> None:
        """

        :param flag:
        :return:
        """
        self.stay_in_current_step = flag

    def get_stay_in_current_step(self) -> bool:
        """

        :return:
        """
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

    def set_run_state(self, state: int) -> None:
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
            self.stepCounter = self.__mcs_direct_access
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

    def read_simulation_data_non_blocking(self, file_number: int) -> bool:
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

    def extract_lattice_description_info(self, file_name: str) -> None:
        """
        Reads the xml that summarizes serialized simulation files
        :param file_name:{str} xml metadata file name
        :return: None
        """

        # lattice data simulation file
        lds_file = os.path.abspath(file_name)
        self.ldsDir = os.path.dirname(lds_file)

        xml2_obj_converter = XMLUtils.Xml2Obj()
        root_element = xml2_obj_converter.Parse(lds_file)
        dim_element = root_element.getFirstElement("Dimensions")
        self.fieldDim = CompuCell.Dim3D()
        self.fieldDim.x = int(dim_element.getAttribute("x"))
        self.fieldDim.y = int(dim_element.getAttribute("y"))
        self.fieldDim.z = int(dim_element.getAttribute("z"))
        output_element = root_element.getFirstElement("Output")
        self.ldsCoreFileName = output_element.getAttribute("CoreFileName")
        self.frequency = int(output_element.getAttribute("Frequency"))
        self.numberOfSteps = int(output_element.getAttribute("NumberOfSteps"))

        # obtaining list of files in the ldsDir
        lattice_element = root_element.getFirstElement("Lattice")
        self.latticeType = lattice_element.getAttribute("Type")

        # getting information about cell type names and cell ids.
        # It is necessary during generation of the PIF files from VTK output
        cell_types_elements = root_element.getElements("CellType")
        list_cell_type_elements = CC3DXMLListPy(cell_types_elements)
        for cell_type_element in list_cell_type_elements:
            type_name = cell_type_element.getAttribute("TypeName")
            type_id = cell_type_element.getAttributeAsInt("TypeId")
            self.typeIdTypeNameDict[type_id] = type_name

        # now will convert python dictionary into C++ map<int, string>     

        self.typeIdTypeNameCppMap = CC3DXML.MapIntStr()
        for type_id in list(self.typeIdTypeNameDict.keys()):
            self.typeIdTypeNameCppMap[int(type_id)] = self.typeIdTypeNameDict[type_id]

        lds_file_list = os.listdir(self.ldsDir)

        for fName in lds_file_list:
            if re.match(".*\.vtk$", fName):
                self.ldsFileList.append(fName)

        self.ldsFileList.sort()

        # extracting information about fields in the lds file
        fields_element = root_element.getFirstElement("Fields")
        if fields_element:
            field_list = XMLUtils.CC3DXMLListPy(fields_element.getElements("Field"))

            for field_elem in field_list:

                field_name = field_elem.getAttribute("Name")
                self.fieldsUsed[field_elem.getAttribute("Name")] = field_elem.getAttribute("Type")
                if field_elem.findAttribute("Script"):  # True or False if present
                    # ToDo:  if a "CustomVis" Type was provided,
                    #  require that a "Script" was also provided; else warn user
                    custom_vis_script = field_elem.getAttribute("Script")

                    self.customVis = CompuCellSetup.createCustomVisPy(field_name)
                    self.customVis.registerVisCallbackFunction(CompuCellSetup.vtkScriptCallback)

                    self.customVis.addActor(field_name, "vtkActor")
                    # we'll piggyback off the actorsDict
                    self.customVis.addActor("customVTKScript", custom_vis_script)

    def generate_pif_from_vtk(self, _vtkFileName: str, _pifFileName: str) -> None:
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

    def set_current_step_direct_access(self, mcs: int) -> None:
        """
        function used by lattice files data panel module to directly set
        current step
        :param mcs: {int} current mcs - directo access from gui pabel
        :return:
        """
        self.__mcs_direct_access = mcs
        self.__direct_access_flag = True

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
