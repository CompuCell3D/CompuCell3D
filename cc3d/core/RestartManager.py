# -*- coding: utf-8 -*-
import os
import pickle
import json
from collections import OrderedDict
from cc3d.cpp import CompuCell
from cc3d.cpp import SerializerDEPy
from cc3d.core.PySteppables import CellList
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.core import XMLUtils
import cc3d
from cc3d import CompuCellSetup
from .CC3DSimulationDataHandler import CC3DSimulationDataHandler
from .SteeringParam import SteeringParam
from pathlib import Path
import shutil
import copyreg


def pickle_vector3(_vec):
    return CompuCell.Vector3, (_vec.fX, _vec.fY, _vec.fZ)


copyreg.pickle(CompuCell.Vector3, pickle_vector3)


class RestartManager:

    def __init__(self, _sim=None):

        sim = CompuCellSetup.persistent_globals.simulator
        self.serializer = SerializerDEPy.SerializerDE()

        self.serializer.init(sim)

        self.cc3dSimOutputDir = ''
        self.serializeDataList = []
        # field size for formatting step number output
        self.__step_number_of_digits = 0
        self.__completedRestartOutputPath = ''
        self.allow_multiple_restart_directories = False
        self.output_frequency = 0
        self.__baseSimulationFilesCopied = False

        # variables used during restarting
        self.__restartDirectory = ''
        self.__restart_file = ''
        self.__restartVersion = 0
        self.__restartBuild = 0
        self.__restart_step = 0
        self.__restart_resource_dict = {}

        self.cc3d_simulation_data_handler = None

    def get_restart_step(self):
        return self.__restart_step

    def prepare_restarter(self):
        """
        Performs basic setup before attempting a restart
        :return: None
        """

        pg = CompuCellSetup.persistent_globals

        self.cc3d_simulation_data_handler = CC3DSimulationDataHandler()
        if pg.simulation_file_name:
            self.cc3d_simulation_data_handler.read_cc3_d_file_format(pg.simulation_file_name)

    @staticmethod
    def restart_enabled():
        """
        reads .cc3d project file and checks if restart is enabled
        :return: {bool}
        """
        sim_file_name = CompuCellSetup.persistent_globals.simulation_file_name

        return sim_file_name is not None and Path(sim_file_name).parent.joinpath('restart').exists()

    @staticmethod
    def append_xml_stub(root_elem, sd):
        """
        Internal function in the restart manager - manipulatex xml file that describes the layout of
        restart files
        :param root_elem: {instance of CC3DXMLElement}
        :param sd: {object that has basic information about serialized module}
        :return: None
        """
        base_file_name = os.path.basename(sd.fileName)
        attribute_dict = {"ModuleName": sd.moduleName, "ModuleType": sd.moduleType, "ObjectName": sd.objectName,
                         "ObjectType": sd.objectType, "FileName": base_file_name, 'FileFormat': sd.fileFormat}
        root_elem.ElementCC3D('ObjectData', attribute_dict)

    @staticmethod
    def get_restart_output_root_path(restart_output_path):
        """
        returns path to the  output root directory e.g. <outputFolder>/restart_200
        :param restart_output_path: {str}
        :return:{str}
        """
        return str(Path(restart_output_path).parent)

    def setup_restart_output_directory(self, step=0):
        """
        Prepares restart directory
        :param step: {int} Monte Carlo Step
        :return: {None}
        """

        pg = CompuCellSetup.persistent_globals
        output_dir_root = pg.output_directory
        if not self.__step_number_of_digits:
            self.__step_number_of_digits = len(str(pg.simulator.getNumSteps()))

        restart_output_root = Path(output_dir_root).joinpath(
            'restart_' + str(step).zfill(self.__step_number_of_digits))
        restart_files_dir = restart_output_root.joinpath('restart')

        restart_files_dir.mkdir(parents=True, exist_ok=True)
        pg.copy_simulation_files_to_output_folder(custom_output_directory=restart_output_root)


        return str(restart_files_dir)

    def updatePythonScript(self, _fileName):
        """
        Manipulates Python script - alters the content to make sure it is restart -ready
        :param _fileName: {str} path to Python file
        :return: None
        """
        if _fileName == '':
            return

        import re
        dimRegex = re.compile('([\s\S]*.ElementCC3D\([\s]*"Dimensions")([\S\s]*)(\)[\s\S]*)')
        commentRegex = re.compile('^([\s]*#)')

        try:
            fXMLNew = open(_fileName + '.new', 'w')
        except IOError as e:
            print(__file__ + ' updatePythonScript: could not open ', _fileName, ' for writing')

        fieldDim = self.sim.getPotts().getCellFieldG().getDim()

        for line in open(_fileName):
            lineTmp = line.rstrip()
            groups = dimRegex.search(lineTmp)

            commentGroups = commentRegex.search(lineTmp)
            if commentGroups:
                print(line.rstrip(), file=fXMLNew)
                continue

            if groups and groups.lastindex == 3:
                dimString = ',{"x":' + str(fieldDim.x) + ',' + '"y":' + str(fieldDim.y) + ',' + '"z":' + str(
                    fieldDim.z) + '}'
                newLine = dimRegex.sub(r'\1' + dimString + r'\3', lineTmp)
                print(newLine, file=fXMLNew)
            else:
                print(line.rstrip(), file=fXMLNew)

        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName + '.new', _fileName)

    def updateXMLScript(self, _fileName=''):

        """
        Manipulates XML script - alters the content to make sure it is restart -ready
        :param _fileName: {str} path to XML file
        :return: None
        """

        if _fileName == '':
            return

        import re
        dimRegex = re.compile('([\s]*<Dimensions)([\S\s]*)(/>[\s]*)')

        try:
            fXMLNew = open(_fileName + '.new', 'w')
        except IOError as e:
            print(__file__ + ' updateXMLScript: could not open ', _fileName, ' for writing')

        fieldDim = self.sim.getPotts().getCellFieldG().getDim()
        for line in open(_fileName):
            lineTmp = line.rstrip()
            groups = dimRegex.search(lineTmp)

            if groups and groups.lastindex == 3:
                dimString = ' x="' + str(fieldDim.x) + '" ' + 'y="' + str(fieldDim.y) + '" ' + 'z="' + str(
                    fieldDim.z) + '" '
                newLine = dimRegex.sub(r'\1' + dimString + r'\3', lineTmp)
                print(newLine, file=fXMLNew)
            else:

                print(line.rstrip(), file=fXMLNew)

        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName + '.new', _fileName)

    def read_restart_file(self, _fileName):
        """
        reads XML file that holds information about restart data
        :param _fileName: {str}
        :return: None
        """
        xml2_obj_converter = XMLUtils.Xml2Obj()

        file_full_path = os.path.abspath(_fileName)

        root_element = xml2_obj_converter.Parse(file_full_path)  # this is RestartFiles element
        if root_element.findAttribute('Version'):
            self.__restartVersion = root_element.getAttribute('Version')
        if root_element.findAttribute('Build'):
            self.__restartVersion = root_element.getAttributeAsInt('Build')

        step_elem = root_element.getFirstElement('Step')

        if step_elem:
            self.__restart_step = step_elem.getInt()

        restart_object_elements = XMLUtils.CC3DXMLListPy(root_element.getElements('ObjectData'))

        if restart_object_elements:
            for elem in restart_object_elements:
                sd = SerializerDEPy.SerializeData()
                if elem.findAttribute('ObjectName'):
                    sd.objectName = elem.getAttribute('ObjectName')
                if elem.findAttribute('ObjectType'):
                    sd.objectType = elem.getAttribute('ObjectType')
                if elem.findAttribute('ModuleName'):
                    sd.moduleName = elem.getAttribute('ModuleName')
                if elem.findAttribute('ModuleType'):
                    sd.moduleType = elem.getAttribute('ModuleType')
                if elem.findAttribute('FileName'):
                    sd.fileName = elem.getAttribute('FileName')
                if elem.findAttribute('FileFormat'):
                    sd.fileFormat = elem.getAttribute('FileFormat')
                if sd.objectName != '':
                    self.__restart_resource_dict[sd.objectName] = sd
        print('self.__restartResourceDict=', self.__restart_resource_dict)

    def loadRestartFiles(self):
        """
        Loads restart files
        :return: None
        """

        pg = CompuCellSetup.persistent_globals
        restart_file_pth = Path(pg.simulation_file_name).parent.joinpath('restart', 'restart.xml')

        if not restart_file_pth.exists():
            return

        self.__restart_file = str(restart_file_pth)
        self.__restartDirectory = str(restart_file_pth.parent)
        self.read_restart_file(self.__restart_file)

        # if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
        #
        #     print("EXTRACTING restartEnabled")
        #     from . import CC3DSimulationDataHandler
        #
        #     cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler()
        #     cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
        #     print("cc3dSimulationDataHandler.cc3dSimulationData.serializerResource=",
        #           cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory)
        #     if cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory != '':
        #         restartFileLocation = os.path.dirname(str(CompuCellSetup.simulationFileName))
        #         self.__restartDirectory = os.path.join(restartFileLocation, 'restart')
        #         self.__restartDirectory = os.path.abspath(self.__restartDirectory)  # normalizing path format
        #
        #         self.__restart_file = os.path.join(self.__restartDirectory, 'restart.xml')
        #         print('self.__restartDirectory=', self.__restartDirectory)
        #         print('self.__restartFile=', self.__restart_file)
        #         self.read_restart_file(self.__restart_file)

        # ---------------------- LOADING RESTART FILES    --------------------
        # loading cell field    

        self.load_cell_field()

        # loading concentration fields (scalar fields) from PDE solvers            
        self.load_concentration_fields()

        # loading extra scalar fields   - used in Python only
        self.load_scalar_fields()

        # loading extra scalar fields cell level - used in Python only
        self.load_scalar_fields_cell_level()

        # loading shared vector fields numpy - shared between Python and C++
        self.load_shared_vector_fields_numpy()

        # loading extra vector fields  - used in Python only
        self.load_vector_fields()

        # loading extra vector fields cell level - used in Python only
        self.load_vector_fields_cell_level()

        # loading core cell  attributes
        self.load_core_cell_attributes()

        # load cell Python attributes
        self.load_python_attributes()

        # load SBMLSolvers -  free floating SBML Solvers are loaded and initialized
        # and those associated with cell are initialized - they are loaded by  self.loadPythonAttributes
        self.load_sbml_solvers()

        # load adhesionFlex plugin
        self.load_adhesion_flex()

        # load chemotaxis plugin        
        self.load_chemotaxis()

        # load LengthConstraint plugin        
        self.load_length_constraint()

        # load ConnectivityGlobal plugin        
        self.load_connectivity_global()

        # load ConnectivityLocalFlex plugin        
        self.load_connectivity_local_flex()

        # load FocalPointPlasticity plugin        
        self.load_focal_point_plasticity()

        # load ContactLocalProduct plugin        
        self.load_contact_local_product()

        # load CellOrientation plugin        
        self.load_cell_orientation()

        # load PolarizationVector plugin        
        self.load_polarization_vector()

        # load loadPolarization23 plugin        
        self.load_polarization23()

        # load steering panel
        self.load_steering_panel()

        # ---------------------- END OF LOADING RESTART FILES    --------------------

    def load_cell_field(self, ):
        """
        Restores Cell Field
        :return: None
        """
        if 'CellField' in list(self.__restart_resource_dict.keys()):
            sd = self.__restart_resource_dict['CellField']
            # full path to cell field serialized resource
            full_path = os.path.join(self.__restartDirectory, sd.fileName)
            full_path = os.path.abspath(full_path)  # normalizing path format
            tmp_file_name = sd.fileName
            sd.fileName = full_path
            self.serializer.loadCellField(sd)
            sd.fileName = tmp_file_name

    def load_concentration_fields(self):
        """
        restores chemical fields
        :return: None
        """

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectType == 'ConcentrationField':
                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path
                self.serializer.loadConcentrationField(sd)
                sd.fileName = tmp_file_name

    def load_scalar_fields(self):
        """
        restores user-defined custom scalar fields (not associated with PDE solvers)
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry

        scalar_fields_dict = field_registry.getScalarFields()
        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectType == 'ScalarField' and sd.moduleType == 'Python':

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path

                try:
                    sd.objectPtr = scalar_fields_dict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadScalarField(sd)
                sd.fileName = tmp_file_name

    def load_scalar_fields_cell_level(self):
        """
        Loads user-defined custom scalar fields (not associated with PDE solvers) that are defined on per-cell basis
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry

        scalar_fields_dict_cell_level = field_registry.getScalarFieldsCellLevel()
        for resource_name, sd in self.__restart_resource_dict.items():

            if sd.objectType == 'ScalarFieldCellLevel' and sd.moduleType == 'Python':

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path

                try:
                    sd.objectPtr = scalar_fields_dict_cell_level[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadScalarFieldCellLevel(sd)
                sd.fileName = tmp_file_name

    def load_shared_vector_fields_numpy(self):
        """
        restores user-defined custom vector fields
        :return: None
        """

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectType == 'SharedVectorFieldNumpy':
                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path
                self.serializer.loadSharedVectorFieldNumpy(sd)
                sd.fileName = tmp_file_name


    def load_vector_fields(self):
        """
        restores user-defined custom vector fields
        :return: None
        """

        field_registry = CompuCellSetup.persistent_globals.field_registry
        vector_fields_dict = field_registry.getVectorFields()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectType == 'VectorField' and sd.moduleType == 'Python':

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path

                try:
                    sd.objectPtr = vector_fields_dict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadVectorField(sd)
                sd.fileName = tmp_file_name

    def load_vector_fields_cell_level(self):
        """
        Loads user-defined custom vector fields that are defined on per-cell basis
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry

        vector_fields_cell_level_dict = field_registry.getVectorFieldsCellLevel()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectType == 'VectorFieldCellLevel' and sd.moduleType == 'Python':

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                tmp_file_name = sd.fileName
                sd.fileName = full_path

                try:
                    sd.objectPtr = vector_fields_cell_level_dict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadVectorFieldCellLevel(sd)
                sd.fileName = tmp_file_name

    def load_core_cell_attributes(self):
        """
        Loads core cell attributes such as lambdaVolume, targetVolume etc...
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'CoreCellAttributes' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)
                for cell in cell_list:
                    cell_id = pickle.load(pf)
                    cluster_id = cell.clusterId
                    cell_core_attributes = pickle.load(pf)

                    if cell:
                        self.set_cell_core_attributes(cell, cell_core_attributes)

                pf.close()

    def unpickle_dict(self, file_name, cell_list):
        """
        Utility function that unpickles dictionary representing dictionary of attributes that user
        attaches to cells at the Python level
        :param file_name: {str}
        :param cell_list: {container with all CC3D cells - equivalent of self.cellList in SteppableBasePy}
        :return:
        """
        try:
            pf = open(file_name, 'rb')
        except IOError:
            return

        number_of_cells = pickle.load(pf)

        for cell in cell_list:
            cell_id = pickle.load(pf)

            unpickled_attrib_dict = pickle.load(pf)

            dict_attrib = CompuCell.getPyAttrib(cell)

            # dict_attrib=copy.deepcopy(unpickled_attrib_dict)
            # adds all objects from unpickled_attrib_dict to dict_attrib -  note: deep copy will not work here
            dict_attrib.update(unpickled_attrib_dict)

        pf.close()

    def unpickle_list(self, file_name, cell_list):
        """
        Utility function that unpickles list representing list of attributes that user
        attaches to cells at the Python level

        :param file_name: {ste}
        :param cell_list: {container with all CC3D cells - equivalent of self.cellList in SteppableBasePy}
        :return:
        """

        try:
            pf = open(file_name, 'r')
        except IOError as e:
            return

        number_of_cells = pickle.load(pf)

        for cell in cell_list:
            cell_id = pickle.load(pf)
            unpickled_attrib_list = pickle.load(pf)
            list_attrib = CompuCell.getPyAttrib(cell)

            # appends all elements of unpickled_attrib_list to the end of list_attrib
            #  note: deep copy will not work here
            list_attrib.extend(unpickled_attrib_list)

        pf.close()

    def load_python_attributes(self):
        """
        Loads python attributes that user attached to cells (a list or dictionary)
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'PythonAttributes' and sd.objectType == 'Pickle':

                full_path = os.path.join(self.__restartDirectory, sd.fileName)

                # normalizing path format
                full_path = os.path.abspath(full_path)

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                # checking if cells have extra attribute
                for cell in cell_list:
                    if not CompuCell.isPyAttribValid(cell):
                        return

                list_flag = True
                for cell in cell_list:
                    attrib = CompuCell.getPyAttrib(cell)
                    if isinstance(attrib, list):
                        list_flag = True
                    else:
                        list_flag = False
                    break

                if list_flag:
                    self.unpickle_list(full_path, cell_list)
                else:
                    self.unpickle_dict(full_path, cell_list)

    def load_sbml_solvers(self):
        """
        Loads SBML solvers
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        free_floating_sbml_simulators = CompuCellSetup.persistent_globals.free_floating_sbml_simulators

        # loading and initializing freeFloating SBML Simulators
        #  SBML solvers associated with cells are loaded (but not fully initialized) in the loadPythonAttributes
        for resource_name, sd in self.__restart_resource_dict.items():
            print('resource_name=', resource_name)
            print('sd=', sd)

            if sd.objectName == 'FreeFloatingSBMLSolvers' and sd.objectType == 'Pickle':
                print('RESTORING FreeFloatingSBMLSolvers ')

                full_path = os.path.join(self.__restartDirectory, sd.fileName)

                # normalizing path format
                full_path = os.path.abspath(full_path)
                with open(full_path, 'r') as pf:
                    CompuCellSetup.freeFloatingSBMLSimulator = pickle.load(pf)

                # initializing  freeFloating SBML Simulators       
                for model_name, sbml_solver in free_floating_sbml_simulators.items():
                    sbml_solver.loadSBML(_externalPath=sim.getBasePath())

        # full initializing SBML solvers associated with cell
        #  we do that regardless whether we have freeFloatingSBMLSolver pickled file or not
        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)

        # checking if cells have extra attribute
        for cell in cell_list:
            if not CompuCell.isPyAttribValid(cell):
                return
            else:
                attrib = CompuCell.getPyAttrib(cell)
                if isinstance(attrib, list):
                    return
                else:
                    break

        for cell in cell_list:

            cell_dict = CompuCell.getPyAttrib(cell)
            try:
                sbml_dict = cell_dict['SBMLSolver']
                print('sbml_dict=', sbml_dict)
            except LookupError:
                continue

            for model_name, sbml_solver in sbml_dict.items():
                # this call fully initializes SBML Solver by
                # loadSBML
                # ( relative path stored in sbml_solver.path and root dir is passed using self.sim.getBasePath())
                sbml_solver.loadSBML(_externalPath=sim.getBasePath())

    def load_adhesion_flex(self):
        """
        restores AdhesionFlex Plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator

        if not sim.pluginManager.isLoaded("AdhesionFlex"):
            return

        adhesion_flex_plugin = CompuCell.getAdhesionFlexPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'AdhesionFlex' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)

                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'r')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)
                # read medium adhesion molecule vector
                medium_adhesion_vector = pickle.load(pf)

                adhesion_flex_plugin.assignNewMediumAdhesionMoleculeDensityVector(medium_adhesion_vector)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    cell_adhesion_vector = pickle.load(pf)
                    adhesion_flex_plugin.assignNewAdhesionMoleculeDensityVector(cell, cell_adhesion_vector)

                pf.close()
            adhesion_flex_plugin.overrideInitialization()

    def load_chemotaxis(self):
        """
        restores Chemotaxis
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator

        if not sim.pluginManager.isLoaded("Chemotaxis"):
            return
        chemotaxis_plugin = CompuCell.getChemotaxisPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'Chemotaxis' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    # loading number of chemotaxis data that cell has
                    chd_number = pickle.load(pf)

                    for i in range(chd_number):
                        # reading chemotaxis data 
                        chd_dict = pickle.load(pf)
                        # creating chemotaxis data for cell
                        chd = chemotaxis_plugin.addChemotaxisData(cell, chd_dict['fieldName'])
                        chd.setLambda(chd_dict['lambda'])
                        chd.saturationCoef = chd_dict['saturationCoef']
                        chd.setChemotaxisFormulaByName(chd_dict['formulaName'])
                        chd.assignChemotactTowardsVectorTypes(chd_dict['chemotactTowardsTypesVec'])

                pf.close()

    def load_length_constraint(self):
        """
        Restores LengthConstraint
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("LengthConstraint"):
            return

        length_constraint_plugin = CompuCell.getLengthConstraintPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'LengthConstraint' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    length_constraint_vec = pickle.load(pf)
                    length_constraint_plugin.setLengthConstraintData(cell, length_constraint_vec[0],
                                                                     length_constraint_vec[1],
                                                                     length_constraint_vec[2])

                pf.close()

    def load_connectivity_global(self):
        """
        Restores ConnectivityGlobal plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityGlobal"):
            return

        connectivity_global_plugin = CompuCell.getConnectivityGlobalPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'ConnectivityGlobal' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    connectivity_strength = pickle.load(pf)
                    connectivity_global_plugin.setConnectivityStrength(cell, connectivity_strength)

                pf.close()

    def load_connectivity_local_flex(self):
        """
        Restores ConnectivityLocalFlex plugin
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            return

        connectivity_local_flex_plugin = CompuCell.getConnectivityLocalFlexPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'ConnectivityLocalFlex' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    connectivity_strength = pickle.load(pf)
                    connectivity_local_flex_plugin.setConnectivityStrength(cell, connectivity_strength)

                pf.close()

    def load_focal_point_plasticity(self):
        """
        restores FocalPointPlasticity plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("FocalPointPlasticity"):
            return

        focal_point_plasticity_plugin = CompuCell.getFocalPointPlasticityPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'FocalPointPlasticity' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    cell_id = cell.id
                    cluster_id = cell.clusterId

                    # read number of fpp links in the cell (external)
                    links_number = pickle.load(pf)
                    for i in range(links_number):
                        # loading external links
                        fpp_dict = pickle.load(pf)
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        # cell_id, cluster id
                        neighbor_ids = fpp_dict['neighborIds']

                        neighbor_cell = inventory.getCellByIds(neighbor_ids[0], neighbor_ids[1])
                        fpptd.neighborAddress = neighbor_cell
                        fpptd.lambdaDistance = fpp_dict['lambdaDistance']
                        fpptd.targetDistance = fpp_dict['targetDistance']
                        fpptd.maxDistance = fpp_dict['maxDistance']
                        fpptd.activationEnergy = fpp_dict['activationEnergy']
                        fpptd.maxNumberOfJunctions = fpp_dict['maxNumberOfJunctions']
                        fpptd.neighborOrder = fpp_dict['neighborOrder']

                        focal_point_plasticity_plugin.insertFPPData(cell, fpptd)

                    # read number of fpp links in the cell (internal)
                    internal_links_number = pickle.load(pf)
                    for i in range(internal_links_number):
                        # loading external links
                        fpp_dict = pickle.load(pf)
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        # cell_id, cluster id
                        neighbor_ids = fpp_dict['neighbor_ids']
                        neighbor_cell = inventory.getCellByIds(neighbor_ids[0], neighbor_ids[1])
                        fpptd.neighborAddress = neighbor_cell
                        fpptd.lambdaDistance = fpp_dict['lambdaDistance']
                        fpptd.targetDistance = fpp_dict['targetDistance']
                        fpptd.maxDistance = fpp_dict['maxDistance']
                        fpptd.activationEnergy = fpp_dict['activationEnergy']
                        fpptd.maxNumberOfJunctions = fpp_dict['maxNumberOfJunctions']
                        fpptd.neighborOrder = fpp_dict['neighborOrder']
                        focal_point_plasticity_plugin.insertInternalFPPData(cell, fpptd)

                    # read number of fpp links in the cell (anchors)
                    anchor_links_number = pickle.load(pf)
                    for i in range(anchor_links_number):
                        # loading external links
                        fpp_dict = pickle.load(pf)
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        # neighbor_ids=fpp_dict['neighbor_ids'] # cell_id, cluster id
                        # neighbor_cell=inventory.getCellByIds(neighbor_ids[0],neighbor_ids[1])
                        fpptd.neighborAddfess = 0
                        fpptd.lambdaDistance = fpp_dict['lambdaDistance']
                        fpptd.targetDistance = fpp_dict['targetDistance']
                        fpptd.maxDistance = fpp_dict['maxDistance']
                        fpptd.anchorId = fpp_dict['anchorId']
                        fpptd.anchorPoint[0] = fpp_dict['anchorPoint'][0]
                        fpptd.anchorPoint[1] = fpp_dict['anchorPoint'][1]
                        fpptd.anchorPoint[2] = fpp_dict['anchorPoint'][2]

                        focal_point_plasticity_plugin.insertAnchorFPPData(cell, fpptd)

                pf.close()

    def load_contact_local_product(self):
        """
        restores ContactLocalProduct plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ContactLocalProduct"):
            return

        contact_local_product_plugin = CompuCell.getContactLocalProductPlugin()

        for resourceName, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'ContactLocalProduct' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)
                    cadherin_vector = pickle.load(pf)
                    contact_local_product_plugin.setCadherinConcentrationVec(
                        cell, CompuCell.contactproductdatacontainertype(cadherin_vector))

                pf.close()

    def load_cell_orientation(self):
        """
        restores CellOriencation plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("CellOrientation"):
            return

        cell_orientation_plugin = CompuCell.getCellOrientationPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'CellOrientation' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)
                    lambda_cell_orientation = pickle.load(pf)
                    cell_orientation_plugin.setLambdaCellOrientation(cell, lambda_cell_orientation)

                pf.close()

    def load_polarization_vector(self):
        """
        restores polarizationVector plugin
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("PolarizationVector"):
            return

        polarization_vector_plugin = CompuCell.getPolarizationVectorPlugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'PolarizationVector' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                # normalizing path format
                full_path = os.path.abspath(full_path)
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)
                    polarization_vec = pickle.load(pf)
                    polarization_vector_plugin.setPolarizationVector(cell, polarization_vec[0], polarization_vec[1],
                                                                     polarization_vec[2])

                pf.close()

    def load_polarization23(self):
        """
        restores polarization23 plugin
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("Polarization23"):
            return

        polarization23_plugin = CompuCell.getPolarization23Plugin()

        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'Polarization23' and sd.objectType == 'Pickle':

                inventory = sim.getPotts().getCellInventory()
                cell_list = CellList(inventory)

                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format
                try:
                    pf = open(full_path, 'rb')
                except IOError:
                    return

                number_of_cells = pickle.load(pf)

                for cell in cell_list:
                    cell_id = pickle.load(pf)

                    # [fX,fY,fZ]
                    pol_vec = pickle.load(pf)
                    pol_markers = pickle.load(pf)
                    lambda_pol = pickle.load(pf)

                    polarization23_plugin.setPolarizationVector(cell,
                                                                CompuCell.Vector3(pol_vec[0], pol_vec[2], pol_vec[2]))
                    polarization23_plugin.setPolarizationMarkers(cell, pol_markers[0], pol_markers[1])
                    polarization23_plugin.setLambdaPolarization(cell, lambda_pol)

                pf.close()

    def output_steering_panel(self, restart_output_path, rst_xml_elem):
        """
        Outputs steering panel for python parameters
        :param restartOutputPath:{str} path to restart dir
        :param rst_xml_elem: {xml elem obj}
        :return: None
        """

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'SteeringPanel'
        sd.moduleType = 'SteeringPanel'
        sd.objectName = 'SteeringPanel'
        sd.objectType = 'JSON'
        sd.fileName = os.path.join(restart_output_path, 'SteeringPanel' + '.json')

        self.serialize_steering_panel(sd.fileName)

        self.append_xml_stub(rst_xml_elem, sd)

    @staticmethod
    def serialize_steering_panel(fname):
        """
        Serializes steering panel
        :param fname: {str} filename - to serialize steering panel to
        :return: None
        """
        steering_param_dict = CompuCellSetup.persistent_globals.steering_param_dict
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
            json.dump(json_dict, outfile, indent=4)

    def load_steering_panel(self):
        """
        Deserializes steering panel
        :return:None
        """
        for resource_name, sd in self.__restart_resource_dict.items():
            if sd.objectName == 'SteeringPanel' and sd.objectType.lower() == 'json':
                full_path = os.path.join(self.__restartDirectory, sd.fileName)
                full_path = os.path.abspath(full_path)  # normalizing path format

                self.deserialize_steering_panel(fname=full_path)

    @staticmethod
    def deserialize_steering_panel(fname):
        """
        Deserializes steering panel
        :param fname: {str} serialized steering panel filename
        :return: None
        """

        CompuCellSetup.persistent_globals.steering_param_dict = OrderedDict()
        steering_param_dict = CompuCellSetup.persistent_globals.steering_param_dict

        with open(fname) as infile:

            json_dict = json.load(infile)
            for param_name, steering_param_obj_data in json_dict.items():
                try:
                    steering_param_dict[param_name] = SteeringParam(
                        name=param_name,
                        val=steering_param_obj_data['val'],
                        min_val=steering_param_obj_data['min'],
                        max_val=steering_param_obj_data['max'],
                        decimal_precision=steering_param_obj_data['decimal_precision'],
                        enum=steering_param_obj_data['enum'],
                        widget_name=steering_param_obj_data['widget_name']
                    )
                except:
                    print
                    'Could not update steering parameter {} value'.format(param_name)

    def output_restart_files(self, step=0, on_demand=False):
        """
        main function that serializes simulation
        :param step: {int} current MCS
        :param on_demand: {False} flag representing whether serialization is ad-hoc or regularly scheduled one
        :return: None
        """

        if not on_demand and self.output_frequency <= 0:
            return

        if not on_demand and step == 0:
            return

        if not on_demand and step % self.output_frequency:
            return

        # have to initialize serialized each time in case lattice gets resized in which case cellField Ptr
        # has to be updated and lattice dimension is usually different

        pg = CompuCellSetup.persistent_globals

        self.serializer.init(pg.simulator)

        rst_xml_elem = ElementCC3D("RestartFiles",
                                   {"Version": cc3d.__version__, 'Build': cc3d.__revision__})
        rst_xml_elem.ElementCC3D("Step", {}, step)
        print('outputRestartFiles')

        # cc3d_sim_output_dir = CompuCellSetup.screenshotDirectoryName
        cc3d_sim_output_dir = pg.output_directory

        print("cc3d_sim_output_dir=", cc3d_sim_output_dir)

        restart_output_path = self.setup_restart_output_directory(step)

        # no output if restart_output_path is not specified
        if restart_output_path == '':
            return

        # ---------------------- OUTPUTTING RESTART FILES    --------------------
        # outputting cell field    
        self.output_cell_field(restart_output_path, rst_xml_elem)

        # outputting concentration fields (scalar fields) from PDE solvers    
        self.output_concentration_fields(restart_output_path, rst_xml_elem)

        # outputting extra scalar fields   - used in Python only
        self.output_scalar_fields(restart_output_path, rst_xml_elem)

        # outputting extra scalar fields cell level  - used in Python only
        self.output_scalar_fields_cell_level(restart_output_path, rst_xml_elem)

        # outputting shared vector fields numpy  - shared between python and C++
        self.output_shared_vector_numpy_fields(restart_output_path, rst_xml_elem)

        # outputting extra vector fields  - used in Python only
        self.output_vector_fields(restart_output_path, rst_xml_elem)

        # outputting extra vector fields cell level  - used in Python only
        self.output_vector_fields_cell_level(restart_output_path, rst_xml_elem)

        # outputting core cell  attributes
        self.output_core_cell_attributes(restart_output_path, rst_xml_elem)

        # outputting cell Python attributes
        self.output_python_attributes(restart_output_path, rst_xml_elem)

        # outputting FreeFloating SBMLSolvers -
        # notice that SBML solvers assoaciated with a cell are pickled in the outputPythonAttributes function
        self.output_free_floating_sbml_solvers(restart_output_path, rst_xml_elem)

        # outputting plugins

        # outputting AdhesionFlexPlugin
        self.output_adhesion_flex_plugin(restart_output_path, rst_xml_elem)

        # outputting ChemotaxisPlugin
        self.output_chemotaxis_plugin(restart_output_path, rst_xml_elem)

        # outputting LengthConstraintPlugin
        self.output_length_constraint_plugin(restart_output_path, rst_xml_elem)

        # outputting ConnectivityGlobalPlugin
        self.output_connectivity_global_plugin(restart_output_path, rst_xml_elem)

        # outputting ConnectivityLocalFlexPlugin
        self.output_connectivity_local_flex_plugin(restart_output_path, rst_xml_elem)

        # outputting FocalPointPlacticityPlugin
        self.output_focal_point_placticity_plugin(restart_output_path, rst_xml_elem)

        # outputting ContactLocalProductPlugin
        self.output_contact_local_product_plugin(restart_output_path, rst_xml_elem)

        # outputting CellOrientationPlugin
        self.output_cell_orientation_plugin(restart_output_path, rst_xml_elem)

        # outputting PolarizationVectorPlugin
        self.output_polarization_vector_plugin(restart_output_path, rst_xml_elem)

        # outputting Polarization23Plugin
        self.output_polarization23_plugin(restart_output_path, rst_xml_elem)
        #
        # # outputting steering panel params
        self.output_steering_panel(restart_output_path, rst_xml_elem)
        #
        # # ---------------------- END OF  OUTPUTTING RESTART FILES    --------------------
        #
        # -------------writing xml description of the restart files
        rst_xml_elem.CC3DXMLElement.saveXML(os.path.join(restart_output_path, 'restart.xml'))

        # --------------- depending on removePreviousFiles we will remove or keep previous restart files

        print('\n\n\n\n self.__allowMultipleRestartDirectories=', self.allow_multiple_restart_directories)

        if not self.allow_multiple_restart_directories:

            print('\n\n\n\n self.__completedRestartOutputPath=', self.__completedRestartOutputPath)

            if self.__completedRestartOutputPath != '':

                try:
                    shutil.rmtree(self.__completedRestartOutputPath)
                except:
                    # will ignore exceptions during directory removal -
                    # they might be due e.g. user accessing directory to be removed -
                    # in such a case it is best to ignore such requests
                    pass

        self.__completedRestartOutputPath = self.get_restart_output_root_path(restart_output_path)

    def output_concentration_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes concentration fields (associated with PDE solvers)
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        conc_field_name_vec = sim.getConcentrationFieldNameVector()
        for fieldName in conc_field_name_vec:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'PDESolver'
            sd.moduleType = 'Steppable'

            sd.objectName = fieldName
            sd.objectType = 'ConcentrationField'
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            print('sd.fileName=', sd.fileName)
            sd.fileFormat = 'text'
            self.serializeDataList.append(sd)
            self.serializer.serializeConcentrationField(sd)
            self.append_xml_stub(rst_xml_elem, sd)
            print("Got concentration field: ", fieldName)

        # serialize generic concentration fields - different precision
        generic_conc_field_name_vec = sim.getGenericScalarFieldNameVectorEngineOwned()
        for fieldName in generic_conc_field_name_vec:
            print("\n\n\nGENERIC FIELD=",fieldName)

            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'PDESolver'
            sd.moduleType = 'Steppable'

            sd.objectName = fieldName
            sd.objectType = 'ConcentrationField'
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            print('sd.fileName=', sd.fileName)
            sd.fileFormat = 'text'
            self.serializeDataList.append(sd)
            self.serializer.serializeConcentrationField(sd)

            self.append_xml_stub(rst_xml_elem, sd)
            print("Got concentration field: ", fieldName)


    def output_cell_field(self, restart_output_path, rst_xml_elem):
        """
        Serializes cell field
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        concFieldNameVec = sim.getConcentrationFieldNameVector()
        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Potts3D'
        sd.moduleType = 'Core'

        sd.objectName = 'CellField'
        sd.objectType = 'CellField'
        sd.fileName = os.path.join(restart_output_path, sd.objectName + '.dat')
        sd.fileFormat = 'text'
        self.serializeDataList.append(sd)
        self.serializer.serializeCellField(sd)
        self.append_xml_stub(rst_xml_elem, sd)

    def output_scalar_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers)
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry

        scalar_fields_dict = field_registry.getScalarFields()
        for fieldName in scalar_fields_dict:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'ScalarField'
            sd.objectPtr = scalar_fields_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeScalarField(sd)
            self.append_xml_stub(rst_xml_elem, sd)

    def output_scalar_fields_cell_level(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers) that are
        defined on the per-cell basis
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry
        scalar_fields_dict_cell_level = field_registry.getScalarFieldsCellLevel()
        for fieldName in scalar_fields_dict_cell_level:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'ScalarFieldCellLevel'
            sd.objectPtr = scalar_fields_dict_cell_level[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeScalarFieldCellLevel(sd)
            self.append_xml_stub(rst_xml_elem, sd)

    def output_shared_vector_numpy_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes numpy vector fields that are shared between python and the C++ engine
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        vector_field_name_vec = sim.getVectorFieldNameVector()

        for fieldName in vector_field_name_vec:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'SharedVectorFieldNumpy'
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeSharedVectorFieldNumpy(sd)
            self.append_xml_stub(rst_xml_elem, sd)


    def output_vector_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined vector fields
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """

        field_registry = CompuCellSetup.persistent_globals.field_registry
        vector_fields_dict = field_registry.getVectorFields()

        for fieldName in vector_fields_dict:

            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'VectorField'
            sd.objectPtr = vector_fields_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeVectorField(sd)
            self.append_xml_stub(rst_xml_elem, sd)

    def output_vector_fields_cell_level(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined vector fields that are defined on per-cell basis
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry
        vector_fields_cell_level_dict = field_registry.getVectorFieldsCellLevel()
        for fieldName in vector_fields_cell_level_dict:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'VectorFieldCellLevel'
            sd.objectPtr = vector_fields_cell_level_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeVectorFieldCellLevel(sd)
            self.append_xml_stub(rst_xml_elem, sd)

    def cell_core_attributes(self, _cell):
        """
        produces a dictionary containing core CellG attributes
        :param _cell:{instance of CellG object} cc3d cell
        :return: {dict}
        """

        coreAttribDict = {}
        coreAttribDict['targetVolume'] = _cell.targetVolume
        coreAttribDict['lambdaVolume'] = _cell.lambdaVolume
        coreAttribDict['targetSurface'] = _cell.targetSurface
        coreAttribDict['lambdaSurface'] = _cell.lambdaSurface
        coreAttribDict['targetClusterSurface'] = _cell.targetClusterSurface
        coreAttribDict['lambdaClusterSurface'] = _cell.lambdaClusterSurface
        coreAttribDict['type'] = _cell.type
        coreAttribDict['xCOMPrev'] = _cell.xCOMPrev
        coreAttribDict['yCOMPrev'] = _cell.yCOMPrev
        coreAttribDict['zCOMPrev'] = _cell.zCOMPrev
        coreAttribDict['lambdaVecX'] = _cell.lambdaVecX
        coreAttribDict['lambdaVecY'] = _cell.lambdaVecY
        coreAttribDict['lambdaVecZ'] = _cell.lambdaVecZ
        coreAttribDict['flag'] = _cell.flag
        coreAttribDict['fluctAmpl'] = _cell.fluctAmpl

        return coreAttribDict

    def set_cell_core_attributes(self, cell, core_attrib_dict):
        """
        initializes cell attributes
        :param cell: {instance of CellG object} cc3d cell
        :param core_attrib_dict: {dict} dictionry of attributes
        :return:
        """

        for attribName, attribValue in core_attrib_dict.items():

            try:
                setattr(cell, attribName, attribValue)

            except LookupError:
                continue
            except AttributeError:
                continue

    def output_core_cell_attributes(self, restart_output_path, rst_xml_elem):
        """
        Serializes core clel attributes - the ones from CellG C++ object such as lambdaVolume, targetVolume, etc...
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Potts3D'
        sd.moduleType = 'Core'
        sd.objectName = 'CoreCellAttributes'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'CoreCellAttributes' + '.dat')
        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)
        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(self.cell_core_attributes(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def pickleList(self, _fileName, _cellList):
        """
        Utility function for pickling CellList object
        :param _fileName: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """
        numberOfCells = len(_cellList)

        nullFile = open(os.devnull, 'w')
        try:
            pf = open(_fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in _cellList:
            # print 'cell.id=',cell.id
            listAttrib = CompuCell.getPyAttrib(cell)
            listToPickle = []
            # checking which list items are picklable
            for item in listAttrib:
                try:
                    pickle.dump(item, nullFile)
                    listToPickle.append(item)
                except TypeError as e:
                    print("PICKLNG LIST")
                    print(e)
                    pass

            pickle.dump(cell.id, pf)
            pickle.dump(listToPickle, pf)

        nullFile.close()
        pf.close()

    def pickleDictionary(self, _fileName, _cellList):
        """
        Utility function for pickling list of attributes attached to cells by user in the Python script
        :param _fileName: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """

        numberOfCells = len(_cellList)

        nullFile = open(os.devnull, 'wb')
        try:
            pf = open(_fileName, 'wb')
        except IOError as e:
            return

        # --------------------------
        # pt=CompuCell.Vector3(10,11,12)

        # pf1=open('PickleCC3D.dat','w')
        # cPickle.dump(pt,pf1)

        # pf1.close()

        # pf1=open('PickleCC3D.dat','r')

        # content=cPickle.load(pf1)
        # print 'content=',content
        # print 'type(content)=',type(content)
        # pf1.close()
        # --------------------------

        pickle.dump(numberOfCells, pf)

        for cell in _cellList:
            # print 'cell.id=',cell.id
            dictAttrib = CompuCell.getPyAttrib(cell)
            dictToPickle = {}
            # checking which list items are un-pickle-able
            for key in dictAttrib:

                try:
                    pickle.dump(dictAttrib[key], nullFile)
                    dictToPickle[key] = dictAttrib[key]

                except TypeError as e:
                    # catching exceptions that my occur (for various reasons) in un-pickle-able objects
                    # print("key=", key, " cannot be pickled")
                    pass
                except KeyError as e:
                    # catching exceptions that my occur (for various reasons) in un-pickle-able objects
                    # print("key=", key, " cannot be pickled")
                    pass
                except AttributeError as e:
                    # catching exceptions that my occur (for various reasons) in un-pickle-able objects
                    # print("key=", key, " cannot be pickled")
                    pass


            pickle.dump(cell.id, pf)
            pickle.dump(dictToPickle, pf)


        nullFile.close()
        pf.close()

    def output_free_floating_sbml_solvers(self, restart_output_path, rst_xml_elem):

        """
        Outputs free-floating SBML solvers
        :param  restart_output_path: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """

        free_floating_sbml_simulators = CompuCellSetup.persistent_globals.free_floating_sbml_simulators

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'FreeFloatingSBMLSolvers'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'FreeFloatingSBMLSolvers' + '.dat')
        # checking if freeFloatingSBMLSimulator is non-empty
        if free_floating_sbml_simulators:
            with open(sd.fileName, 'w') as pf:
                pickle.dump(free_floating_sbml_simulators, pf)
                self.append_xml_stub(rst_xml_elem, sd)

    def output_python_attributes(self, restart_output_path, rst_xml_elem):
        """
        outputs python attributes that were attached to a cell by the user in the Python script
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """
        sim = CompuCellSetup.persistent_globals.simulator
        # notice that this function also outputs SBMLSolver objects
        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)

        # checking if cells have extra attribute

        for cell in cell_list:
            if not CompuCell.isPyAttribValid(cell):
                return

        list_flag = True
        for cell in cell_list:
            attrib = CompuCell.getPyAttrib(cell)
            if isinstance(attrib, list):
                list_flag = True
            else:
                list_flag = False
            break

        print('list_flag=', list_flag)

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'PythonAttributes'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'PythonAttributes' + '.dat')
        # cPickle.dump(numberOfCells,pf)

        if list_flag:
            self.pickleList(sd.fileName, cell_list)
        else:
            self.pickleDictionary(sd.fileName, cell_list)

        self.append_xml_stub(rst_xml_elem, sd)

    def output_adhesion_flex_plugin(self, restart_output_path, rst_xml_elem):
        """
        serializes AdhesionFlex Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("AdhesionFlex"):
            return

        adhesion_flex_plugin = CompuCell.getAdhesionFlexPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'AdhesionFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'AdhesionFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'AdhesionFlex' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)
        # wtiting medium adhesion vector

        medium_adhesion_vector = adhesion_flex_plugin.getMediumAdhesionMoleculeDensityVector()
        pickle.dump(medium_adhesion_vector, pf)
        for cell in cell_list:
            pickle.dump(cell.id, pf)
            cell_adhesion_vector = adhesion_flex_plugin.getAdhesionMoleculeDensityVector(cell)
            pickle.dump(cell_adhesion_vector, pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_chemotaxis_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes Chemotaxis Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("Chemotaxis"):
            return
        chemotaxis_plugin = CompuCell.getChemotaxisPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Chemotaxis'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Chemotaxis'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'Chemotaxis' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)
        for cell in cell_list:
            pickle.dump(cell.id, pf)

            field_names = chemotaxis_plugin.getFieldNamesWithChemotaxisData(cell)
            # outputting numbed of chemotaxis data that cell has
            pickle.dump(len(field_names), pf)

            for fieldName in field_names:
                chd = chemotaxis_plugin.getChemotaxisData(cell, fieldName)
                chd_dict = {}
                chd_dict['fieldName'] = fieldName
                chd_dict['lambda'] = chd.getLambda()
                chd_dict['saturationCoef'] = chd.saturationCoef
                chd_dict['formulaName'] = chd.formulaName
                chemotactTowardsVec = chd.getChemotactTowardsVectorTypes()
                # print('chemotactTowardsVec=', chemotactTowardsVec)
                chd_dict['chemotactTowardsTypesVec'] = chd.getChemotactTowardsVectorTypes()

                pickle.dump(chd_dict, pf)
            # print('field_names=', field_names)
            # cPickle.dump(cellAdhesionVector,pf)        

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_length_constraint_plugin(self, restart_output_path, rst_xml_elem):
        """
        serializes LengthConstraint Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("LengthConstraint"):
            return
        length_constraint_plugin = CompuCell.getLengthConstraintPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'LengthConstraint'
        sd.moduleType = 'Plugin'
        sd.objectName = 'LengthConstraint'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'LengthConstraint' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        lcp = length_constraint_plugin

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump([lcp.getLambdaLength(cell), lcp.getTargetLength(cell), lcp.getMinorTargetLength(cell)], pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_connectivity_global_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ConnectivityGlobal Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityGlobal"):
            return

        connectivity_global_plugin = CompuCell.getConnectivityGlobalPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityGlobal'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityGlobal'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ConnectivityGlobal' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivity_global_plugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_connectivity_local_flex_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ConnectivityLocalFlex Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """
        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            return

        connectivity_local_flex_plugin = CompuCell.getConnectivityLocalFlexPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityLocalFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityLocalFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ConnectivityLocalFlex' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivity_local_flex_plugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_focal_point_placticity_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes FocalPointPlacticity Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("FocalPointPlasticity"):
            return

        focal_point_plasticity_plugin = CompuCell.getFocalPointPlasticityPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'FocalPointPlasticity'
        sd.moduleType = 'Plugin'
        sd.objectName = 'FocalPointPlasticity'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'FocalPointPlasticity' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:

            pickle.dump(cell.id, pf)
            fpp_vec = focal_point_plasticity_plugin.getFPPDataVec(cell)
            internal_fpp_vec = focal_point_plasticity_plugin.getInternalFPPDataVec(cell)
            anchor_fpp_vec = focal_point_plasticity_plugin.getAnchorFPPDataVec(cell)

            # dumping 'external' fpp links
            pickle.dump(len(fpp_vec), pf)
            for fpp_data in fpp_vec:
                fpp_data_dict = {}
                if fpp_data.neighborAddress:
                    fpp_data_dict['neighborIds'] = [fpp_data.neighborAddress.id, fpp_data.neighborAddress.clusterId]
                else:
                    fpp_data_dict['neighborIds'] = [0, 0]
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['activationEnergy'] = fpp_data.activationEnergy
                fpp_data_dict['maxNumberOfJunctions'] = fpp_data.maxNumberOfJunctions
                fpp_data_dict['neighborOrder'] = fpp_data.neighborOrder
                pickle.dump(fpp_data_dict, pf)

            # dumping 'internal' fpp links
            pickle.dump(len(internal_fpp_vec), pf)
            for fpp_data in internal_fpp_vec:
                fpp_data_dict = {}
                if fpp_data.neighborAddress:
                    fpp_data_dict['neighborIds'] = [fpp_data.neighborAddress.id, fpp_data.neighborAddress.clusterId]
                else:
                    fpp_data_dict['neighborIds'] = [0, 0]
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['activationEnergy'] = fpp_data.activationEnergy
                fpp_data_dict['maxNumberOfJunctions'] = fpp_data.maxNumberOfJunctions
                fpp_data_dict['neighborOrder'] = fpp_data.neighborOrder
                pickle.dump(fpp_data_dict, pf)

            # dumping anchor fpp links
            pickle.dump(len(anchor_fpp_vec), pf)
            for fpp_data in anchor_fpp_vec:
                fpp_data_dict = {}
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['anchorId'] = fpp_data.anchorId
                fpp_data_dict['anchorPoint'] = [fpp_data.anchorPoint[0], fpp_data.anchorPoint[1],
                                                fpp_data.anchorPoint[2]]
                pickle.dump(fpp_data_dict, pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_contact_local_product_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ContactLocalProduct Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ContactLocalProduct"):
            return

        contact_local_product_plugin = CompuCell.getContactLocalProductPlugin()
        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ContactLocalProduct'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ContactLocalProduct'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ContactLocalProduct' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(contact_local_product_plugin.getCadherinConcentrationVec(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_cell_orientation_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes CellOrientation Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("CellOrientation"):
            return

        cell_orientation_plugin = CompuCell.getCellOrientationPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'CellOrientation'
        sd.moduleType = 'Plugin'
        sd.objectName = 'CellOrientation'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'CellOrientation' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(cell_orientation_plugin.getLambdaCellOrientation(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_polarization_vector_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes PolarizationVector Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("PolarizationVector"):
            return

        polarization_vector_plugin = CompuCell.getPolarizationVectorPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'PolarizationVector'
        sd.moduleType = 'Plugin'
        sd.objectName = 'PolarizationVector'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'PolarizationVector' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(polarization_vector_plugin.getPolarizationVector(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)

    def output_polarization23_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes Polarization23 Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator

        if not sim.pluginManager.isLoaded("Polarization23"):
            return

        polarization23_plugin = CompuCell.getPolarization23Plugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Polarization23'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Polarization23'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'Polarization23' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pol_vec = polarization23_plugin.getPolarizationVector(cell)
            pickle.dump([pol_vec.fX, pol_vec.fY, pol_vec.fZ], pf)
            pickle.dump(polarization23_plugin.getPolarizationMarkers(cell), pf)
            pickle.dump(polarization23_plugin.getLambdaPolarization(cell), pf)

        pf.close()
        self.append_xml_stub(rst_xml_elem, sd)
