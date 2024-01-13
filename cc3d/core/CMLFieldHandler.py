"""
This class handles field serialization to the (currently) vtk format
"""

from os.path import join
import cc3d.CompuCellSetup
from cc3d.CompuCellSetup.simulation_utils import extract_lattice_type
from cc3d.CompuCellSetup.simulation_utils import extract_type_names_and_ids
from cc3d.core.utils import mkdir_p
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.cpp import PlayerPython


class CMLFieldHandler:

    def __init__(self):

        self.field_storage = PlayerPython.FieldStorage()
        self.field_writer = PlayerPython.FieldWriter()
        self.field_writer.setFieldStorage(self.field_storage)

        # not being used currently
        # self.fieldWriter.setFileTypeToBinary(False)

        self.field_types = {}
        self.output_freq = 1
        self.sim = None
        self.output_dir_name = ""
        self.output_file_core_name = "Step"
        self.out_file_number_of_digits = 0
        self.do_not_output_field_list = []
        self.FIELD_TYPES = (
            "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel")
        self.__initialization_complete = False

    def initialize(self, field_storage=None):
        """
        Initializes CMLFieldHandler
        :param field_storage:
        :return:
        """

        if self.__initialization_complete:
            return

        persistent_globals = cc3d.CompuCellSetup.persistent_globals

        if persistent_globals.output_file_core_name:
            self.output_file_core_name = persistent_globals.output_file_core_name

        self.field_writer.init(persistent_globals.simulator)

        if field_storage is not None:
            self.field_storage = field_storage
            self.field_writer.setFieldStorage(field_storage)

        self.out_file_number_of_digits = len(str(persistent_globals.simulator.getNumSteps()))
        self.get_info_about_fields()

        if cc3d.CompuCellSetup.persistent_globals.output_dir:

            self.create_storage_dir()

            self.write_xml_description_file()

    def write_fields(self,
                     mcs: int,
                     output_dir_name: str = None,
                     output_file_core_name: str = None) -> None:
        """
        stores simulation fields to the disk
        :param mcs: MCS
        :param output_dir_name: override output directory
        :param output_file_core_name: override output file core name
        :return: None
        """

        if output_dir_name is None:
            output_dir_name = self.output_dir_name
        if output_file_core_name is None:
            output_file_core_name = self.output_file_core_name

        if not output_dir_name:
            return

        for field_name in self.field_types.keys():
            if self.field_types[field_name] == self.FIELD_TYPES[0]:
                self.field_writer.addCellFieldForOutput()
            elif self.field_types[field_name] == self.FIELD_TYPES[1]:
                self.field_writer.addConFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[2]:
                self.field_writer.addScalarFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[3]:
                self.field_writer.addScalarFieldCellLevelForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[4]:
                self.field_writer.addVectorFieldForOutput(field_name)
            elif self.field_types[field_name] == self.FIELD_TYPES[5]:
                self.field_writer.addVectorFieldCellLevelForOutput(field_name)

        mcs_formatted_number = str(mcs).zfill(self.out_file_number_of_digits)

        # e.g. /path/Step_01.vtk
        lattice_data_file_name = join(output_dir_name,
                                      output_file_core_name + "_" + mcs_formatted_number + ".vtk")

        self.field_writer.writeFields(lattice_data_file_name)
        self.field_writer.clear()

    def write_xml_description_file(self, file_name: str = "") -> None:
        """
        This function will write XML description of the stored fields. It has to be called after
        initialization of theCMLFieldHandler is completed

        :param file_name:
        :return:
        """

        persistent_globals = cc3d.CompuCellSetup.persistent_globals
        simulator = persistent_globals.simulator

        if not file_name and not self.output_dir_name:
            return

        lattice_type_str = extract_lattice_type()

        if lattice_type_str == '':
            lattice_type_str = 'Square'

        type_id_type_name_dict = extract_type_names_and_ids()

        dim = simulator.getPotts().getCellFieldG().getDim()
        number_of_steps = simulator.getNumSteps()
        lattice_data_xml_element = ElementCC3D("CompuCell3DLatticeData", {"Version": "1.0"})
        lattice_data_xml_element.ElementCC3D("Dimensions", {"x": str(dim.x), "y": str(dim.y), "z": str(dim.z)})
        lattice_data_xml_element.ElementCC3D("Lattice", {"Type": lattice_type_str})
        lattice_data_xml_element.ElementCC3D("Output",
                                             {"Frequency": str(self.output_freq), "NumberOfSteps": str(number_of_steps),
                                              "CoreFileName": self.output_file_core_name,
                                              "Directory": self.output_dir_name})

        # output information about cell type names and cell ids.
        # It is necessary during generation of the PIF files from VTK output
        for typeId in type_id_type_name_dict.keys():
            lattice_data_xml_element.ElementCC3D("CellType",
                                                 {"TypeName": str(type_id_type_name_dict[typeId]),
                                                  "TypeId": str(typeId)})

        fields_xml_element = lattice_data_xml_element.ElementCC3D("Fields")
        for fieldName in self.field_types.keys():
            fields_xml_element.ElementCC3D("Field", {"Name": fieldName, "Type": self.field_types[fieldName]})

        # writing XML description to the disk
        if file_name != "":
            lattice_data_xml_element.CC3DXMLElement.saveXML(str(file_name))
        elif self.output_dir_name:
            lattice_data_file_name = join(self.output_dir_name, self.output_file_core_name + "LDF.dml")
            lattice_data_xml_element.CC3DXMLElement.saveXML(str(lattice_data_file_name))

    def create_storage_dir(self) -> None:
        """
        Creates storage dir for fields. Initializes self.output_dir_name
        :return:
        """
        if cc3d.CompuCellSetup.persistent_globals.simulation_file_name:
            # When a simulation file is present, use PersistentGlobals property that does some fancy work
            screenshot_directory = cc3d.CompuCellSetup.persistent_globals.output_directory
        else:
            # When a simulation file is not present, just use PersistentGlobals property that returns the raw directory
            screenshot_directory = cc3d.CompuCellSetup.persistent_globals.output_dir
            if screenshot_directory is None:
                return
        self.output_dir_name = join(screenshot_directory, 'LatticeData')

        mkdir_p(self.output_dir_name)

    def get_info_about_fields(self) -> None:
        """
        populates a dictionary (self.field_types) with information about currently available
        fields withing the running simulation
        :return: None
        """

        sim = cc3d.CompuCellSetup.persistent_globals.simulator
        # there will always be cell field
        self.field_types["Cell_Field"] = self.FIELD_TYPES[0]

        # extracting information about concentration vectors
        conc_field_name_vec = sim.getConcentrationFieldNameVector()
        for field_name in conc_field_name_vec:
            if field_name not in self.do_not_output_field_list:
                self.field_types[field_name] = self.FIELD_TYPES[1]

        # inserting extra scalar fields managed from Python script
        scalar_field_name_vec = self.field_storage.getScalarFieldNameVector()
        for field_name in scalar_field_name_vec:

            if field_name not in self.do_not_output_field_list:
                self.field_types[field_name] = self.FIELD_TYPES[2]

        # inserting extra scalar fields cell levee managed from Python script
        scalar_field_cell_level_name_vec = self.field_storage.getScalarFieldCellLevelNameVector()
        for field_name in scalar_field_cell_level_name_vec:

            if field_name not in self.do_not_output_field_list:
                self.field_types[field_name] = self.FIELD_TYPES[3]

        # inserting extra vector fields  managed from Python script
        vector_field_name_vec = self.field_storage.getVectorFieldNameVector()
        for field_name in vector_field_name_vec:

            if field_name not in self.do_not_output_field_list:
                self.field_types[field_name] = self.FIELD_TYPES[4]

        # inserting extra vector fields  cell level managed from Python script
        vector_field_cell_level_name_vec = self.field_storage.getVectorFieldCellLevelNameVector()
        for field_name in vector_field_cell_level_name_vec:

            if field_name not in self.do_not_output_field_list:
                self.field_types[field_name] = self.FIELD_TYPES[5]
