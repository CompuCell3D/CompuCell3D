# -*- coding: utf-8 -*-
"""
This module handles  reading of serialized simulation output. It has
API of Simulation Thread and from the point of view of player
it behaves as a "regular" simulation with one exception that
instead running actual simulation during call to the step function it reads previously
saves simulation snapshot (currently stored as vtk file)
"""

from cc3d.cpp import CompuCell
from cc3d.cpp import PlayerPython
from cc3d.core import XMLUtils
import re
from cc3d.core.XMLUtils import CC3DXMLListPy
from cc3d.cpp import CC3DXML
from cc3d.cpp.CompuCell import Dim3D
import cc3d.CompuCellSetup as CompuCellSetup
import vtk
import os
import os.path
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object


def generate_pif_from_vtk(_vtkFileName: str, _pifFileName: str, _lds_file_name: str) -> None:
    """
    generates PIFF from VTK file
    :param _vtkFileName:
    :param _pifFileName:
    :return:
    """
    lds_reader = LatticeDataSummaryReader(_lds_file_name)
    file_writer = PlayerPython.FieldWriter()
    file_writer.generatePIFFileFromVTKOutput(_vtkFileName, _pifFileName,
                                             lds_reader.field_dim.x, lds_reader.field_dim.y, lds_reader.field_dim.z,
                                             lds_reader.type_id_type_name_cpp_map)


class _LatticeDataXMLReader:
    def __init__(self, lds_file: str):
        self.lds_file = os.path.abspath(lds_file)
        assert os.path.isfile(self.lds_file)

        self.xml2_obj_converter = XMLUtils.Xml2Obj()
        self.root_element = self.xml2_obj_converter.Parse(lds_file)
        self.dim_element = self.root_element.getFirstElement("Dimensions")
        self.output_element = self.root_element.getFirstElement("Output")
        self.lattice_element = self.root_element.getFirstElement("Lattice")
        self.cell_types_elements = self.root_element.getElements("CellType")
        self.fields_element = self.root_element.getFirstElement("Fields")


class LatticeDataSummaryReader:
    def __init__(self, lds_file: str):
        """
        Reads the xml that summarizes serialized simulation files
        Instantiate class to import entire summary, or use static methods for specific data
        :param lds_file:{str} xml metadata file name
        """
        self.lds_file = self.lds_file_check(lds_file)
        self.lds_dir = os.path.dirname(lds_file)

        self.field_dim = self.extract_lattice_dim_from_file_name(lds_file)
        self.lds_core_file_name = self.extract_lds_core_file_name_from_file_name(lds_file)
        self.frequency = self.extract_frequency_from_file_name(lds_file)
        self.number_of_steps = self.extract_number_of_steps_from_file_name(lds_file)

        # obtaining list of files in the ldsDir
        self.lattice_type = self.extract_lattice_type_from_file_name(lds_file)

        # getting information about cell type names and cell ids.
        # It is necessary during generation of the PIF files from VTK output
        self.type_id_type_name_dict = self.extract_cell_type_info_from_file_name(lds_file)

        # now will convert python dictionary into C++ map<int, string>

        self.type_id_type_name_cpp_map = self.extract_cell_type_cpp_info_from_file_name(lds_file)

        self.lds_file_list = self.extract_lds_file_list_from_file_name(lds_file)

        # extracting information about fields in the lds file
        self.fields_used = self.extract_fields_used_from_file_name(lds_file)
        self.custom_vis_scripts = self.extract_custom_vis_scripts_from_file_name(lds_file)

    @staticmethod
    def lds_file_check(_lds_file):
        _lds_file = os.path.abspath(_lds_file)
        assert os.path.isfile(_lds_file), f'File not found: {_lds_file}'
        return _lds_file

    @staticmethod
    def extract_lattice_dim_from_file_name(_lds_file) -> Dim3D:
        """
        Extract field dimensions from the xml that summarizes serialized simulation files
        :param _lds_file:{str} xml metadata file name
        :return: {Dim3D} field dimensions
        """
        lds_xml = _LatticeDataXMLReader(_lds_file)

        field_dim = CompuCell.Dim3D()
        field_dim.x = int(lds_xml.dim_element.getAttribute("x"))
        field_dim.y = int(lds_xml.dim_element.getAttribute("y"))
        field_dim.z = int(lds_xml.dim_element.getAttribute("z"))
        return field_dim

    @staticmethod
    def extract_lds_core_file_name_from_file_name(_lds_file: str) -> str:
        """
        Extract core file name from the xml that summarizes serialized simulation files
        :param _lds_file:{str} xml metadata file name
        :return: {str} core file name
        """
        lds_xml = _LatticeDataXMLReader(_lds_file)
        return lds_xml.output_element.getAttribute("CoreFileName")

    @staticmethod
    def extract_frequency_from_file_name(_lds_file: str) -> int:
        lds_xml = _LatticeDataXMLReader(_lds_file)
        return int(lds_xml.output_element.getAttribute("Frequency"))

    @staticmethod
    def extract_number_of_steps_from_file_name(_lds_file: str) -> int:
        lds_xml = _LatticeDataXMLReader(_lds_file)
        return int(lds_xml.output_element.getAttribute("NumberOfSteps"))

    @staticmethod
    def extract_lattice_type_from_file_name(_lds_file: str) -> str:
        lds_xml = _LatticeDataXMLReader(_lds_file)
        return lds_xml.lattice_element.getAttribute("Type")

    @staticmethod
    def extract_cell_type_info_from_file_name(_lds_file: str) -> dict:
        lds_xml = _LatticeDataXMLReader(_lds_file)
        cte = CC3DXMLListPy(lds_xml.cell_types_elements)
        return {el.getAttributeAsInt("TypeId"): el.getAttribute("TypeName") for el in cte}

    @staticmethod
    def extract_cell_type_cpp_info_from_file_name(_lds_file: str) -> CC3DXML.MapIntStr:
        _lds_file = LatticeDataSummaryReader.lds_file_check(_lds_file)
        type_id_type_name_dict = LatticeDataSummaryReader.extract_cell_type_info_from_file_name(_lds_file)
        type_id_type_name_cpp_map = CC3DXML.MapIntStr(
            {int(type_id):type_id_type_name_dict[type_id] for type_id in list(type_id_type_name_dict.keys()) }
        )

        return type_id_type_name_cpp_map

    @staticmethod
    def extract_lds_file_list_from_file_name(_lds_file: str) -> list:
        _lds_dir = os.path.dirname(LatticeDataSummaryReader.lds_file_check(_lds_file))
        lds_file_list = [fName for fName in os.listdir(_lds_dir) if re.match(".*\.vtk$", fName)]
        lds_file_list.sort()
        return lds_file_list

    @staticmethod
    def extract_fields_used_from_file_name(_lds_file: str) -> dict:
        lds_xml = _LatticeDataXMLReader(_lds_file)

        fields_used = dict()
        if lds_xml.fields_element:
            field_list = XMLUtils.CC3DXMLListPy(lds_xml.fields_element.getElements("Field"))
            fields_used.update({el.getAttribute("Name"): el.getAttribute("Type") for el in field_list})
        return fields_used

    @staticmethod
    def extract_custom_vis_scripts_from_file_name(_lds_file: str) -> dict:
        lds_xml = _LatticeDataXMLReader(_lds_file)

        o = dict()
        # ToDo:  if a "CustomVis" Type was provided,
        #  require that a "Script" was also provided; else warn user
        if lds_xml.fields_element:
            fl = XMLUtils.CC3DXMLListPy(lds_xml.fields_element.getElements("Field"))
            o.update({el.getAttribute("Name"): el.getAttribute("Script") for el in fl if el.findAttribute("Script")})
        return o


class CMLResultReader:
    def __init__(self, parent=None, file_name: str = None):
        # If initializing from file_name, field extractor is initialized here
        # Otherwise, a parent must be passed that, at some point before reading simulation data,
        # has a CML reader attached

        # Support pass when runnnig replay in Player
        if parent is None and file_name is None:
            return

        # Will extract fieldExtractor from parent or initialize new fieldExtractor from file_name
        assert file_name is not None or hasattr(parent, 'fieldExtractor')
        self._parent = parent
        self.field_extractor = None
        if parent is None:
            assert os.path.isfile(os.path.abspath(file_name))
            assert file_name.endswith('.dml'), 'file_name must specify a .dml file'
            self.field_extractor = PlayerPython.FieldExtractorCML()
            self.field_extractor.setFieldDim(LatticeDataSummaryReader.extract_lattice_dim_from_file_name(file_name))
            self.field_extractor.setLatticeType(LatticeDataSummaryReader.extract_lattice_type_from_file_name(file_name))

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
        self.typeIdTypeNameDict = {}
        self.__fileWriter = None

        # C++ map<int,string> providing mapping from type id to type name
        self.typeIdTypeNameCppMap = None
        self.customVis = None

    def dimension_change(self) -> bool:
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

    def dimensionChange(self) -> bool:
        return self.dimension_change()

    def __check_field_extractor(self):
        # By this point, either field extractor is set, or parent has cml field extractor
        if self.field_extractor is None:
            assert hasattr(self._parent, 'fieldExtractor') or hasattr(self._parent, 'field_extractor')
            if hasattr(self._parent, 'field_extractor'):
                _a = 'field_extractor'
            else:
                _a = 'fieldExtractor'
            self.field_extractor = getattr(self._parent, _a)
            assert isinstance(self.field_extractor, PlayerPython.FieldExtractorCML)

    def read_simulation_data_from_file(self, vtk_file_name: str) -> (bool, int):
        """
        reads content of the vtk file
        :param vtk_file_name: {str} vtk file name
        :return: {(bool, int)} flag whether read was successful or not, extracted mcs
        """
        vtk_file_name = os.path.abspath(vtk_file_name)
        try:
            assert os.path.isfile(vtk_file_name)
        except AssertionError:
            print(f'File does not exist: {vtk_file_name}')
            return False, -1

        try:
            file_number = self.ldsFileList.index(vtk_file_name)
        except ValueError:
            print(f'File not found in currently loaded directory: {vtk_file_name}')
            return False, -1

        return self.read_simulation_data(file_number)

    def read_simulation_data(self, file_number: int) -> (bool, int):
        """
        reads content of the serialized file
        :param file_number: {int}
        :return: {(bool, int)} flag whether read was successful or not, extracted mcs
        """
        # By this point, either field extractor is set, or parent has cml field extractor
        if self.field_extractor is None:
            assert hasattr(self._parent, 'fieldExtractor') or hasattr(self._parent, 'field_extractor')
            if hasattr(self._parent, 'field_extractor'):
                _a = 'field_extractor'
            else:
                _a = 'fieldExtractor'
            self.field_extractor = getattr(self._parent, _a)
            assert isinstance(self.field_extractor, PlayerPython.FieldExtractorCML)

        # when new data is read from hard drive
        if file_number >= len(self.ldsFileList):
            return False, -1

        fileName = self.ldsFileList[file_number]

        self.simulationDataReader = vtk.vtkStructuredPointsReader()

        self.currentFileName = os.path.join(self.ldsDir, fileName)
        print('self.currentFileName=', self.currentFileName)

        extracted_mcs = self.extract_mcs_number_from_file_name(file_name=self.currentFileName)
        if extracted_mcs < 0:
            extracted_mcs = file_number

        self.simulationDataReader.SetFileName(self.currentFileName)

        data_reader_int_addr = extract_address_int_from_vtk_object(vtkObj=self.simulationDataReader)

        # swig wrapper  on top of     vtkStructuredPointsReader.Update()  - releases GIL,
        # hence can be used in multithreaded program that does not block GUI
        self.field_extractor.readVtkStructuredPointsData(data_reader_int_addr)

        # # # self.simulationDataReader.Update() # does not erlease GIL - VTK ISSUE -
        # limited use in multithreaded program - blocks GUI
        self.simulationData = self.simulationDataReader.GetOutput()

        self.fieldDimPrevious = self.fieldDim

        dim_from_vtk = self.simulationData.GetDimensions()

        self.fieldDim = CompuCell.Dim3D(dim_from_vtk[0], dim_from_vtk[1], dim_from_vtk[2])

        return True, extracted_mcs

    @staticmethod
    def extract_mcs_number_from_file_name(file_name: str) -> int:
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

    @staticmethod
    def extract_lattice_dim_from_file_name(file_name: str) -> Dim3D:
        """
        Extract field dimensions from the xml that summarizes serialized simulation files
        :param file_name:{str} xml metadata file name
        :return: {Dim3D} field dimensions
        """
        return LatticeDataSummaryReader.extract_lattice_dim_from_file_name(file_name)

    def extract_lattice_description_info(self, file_name: str) -> None:
        """
        Reads the xml that summarizes serialized simulation files
        :param file_name:{str} xml metadata file name
        :return: None
        """
        # lattice data simulation file
        lds_file = os.path.abspath(file_name)
        lds_reader = LatticeDataSummaryReader(lds_file=lds_file)

        self.ldsDir = lds_reader.lds_dir
        self.fieldDim = lds_reader.field_dim
        self.ldsCoreFileName = lds_reader.lds_core_file_name
        self.frequency = lds_reader.frequency
        self.numberOfSteps = lds_reader.number_of_steps
        self.latticeType = lds_reader.lattice_type
        self.typeIdTypeNameDict = lds_reader.type_id_type_name_dict
        self.typeIdTypeNameCppMap = lds_reader.type_id_type_name_cpp_map
        self.ldsFileList = lds_reader.lds_file_list
        self.fieldsUsed = lds_reader.fields_used

        for field_name, custom_vis_script in lds_reader.custom_vis_scripts.items():
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
