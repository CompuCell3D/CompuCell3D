import itertools
import numpy as np
from collections import OrderedDict
from cc3d.core.iterators import *
from cc3d.core.enums import *
from cc3d.core.ExtraFieldAdapter import ExtraFieldAdapter
# from cc3d.CompuCellSetup.simulation_utils import stop_simulation
from cc3d.CompuCellSetup.simulation_utils import extract_type_names_and_ids
from cc3d import CompuCellSetup
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
from cc3d.core.XMLDomUtils import XMLElemAdapter
from typing import Union
from cc3d.cpp import CompuCell
from cc3d.core.SBMLSolverHelper import SBMLSolverHelper
import types
import warnings
from deprecated import deprecated
from cc3d.core.SteeringParam import SteeringParam
from copy import deepcopy


class SteppablePy:
    def __init__(self):
        self.runBeforeMCS = 0

    def core_init(self):
        """

        :return:
        """

    def start(self):
        """

        :return:
        """

    def step(self, mcs):
        """

        :param _mcs:
        :return:
        """

    def finish(self):
        """

        :return:
        """

    def cleanup(self):
        """

        :return:
        """


class FieldFetcher:
    def __init__(self):
        """

        """

    def __getattr__(self, item):
        pg = CompuCellSetup.persistent_globals
        field_registry = pg.field_registry

        if item.lower() in ['cell_field', 'cellfield']:
            return pg.simulator.getPotts().getCellFieldG()

        try:
            return field_registry.get_field_adapter(field_name=item)
        except KeyError:
            # trying C++ concentration fields - e.g. diffusion solvers
            field = CompuCell.getConcentrationField(pg.simulator, item)
            if field is not None:
                return field
            else:
                raise KeyError(' The requested field {} does not exist'.format(item))


class GlobalSBMLFetcher:
    def __init__(self):
        """

        """

    def __getattr__(self, item):

        pg = CompuCellSetup.persistent_globals

        item_to_search = item
        rr_flag = False
        if item.startswith('_rr_'):
            item_to_search = item[4:]
            rr_flag = True

        try:

            rr_object = pg.free_floating_sbml_simulators[item_to_search]

        except (LookupError, KeyError):
            raise KeyError('Could not find "free floating" SBML solver with id={sbml_solver_id}'.format(
                sbml_solver_id=item_to_search))

        if rr_flag:
            return rr_object
        else:
            return rr_object.model


class SteppableBasePy(SteppablePy, SBMLSolverHelper):
    (CC3D_FORMAT, TUPLE_FORMAT) = range(0, 2)

    def __init__(self, *args, **kwds):

        try:
            frequency = args[1]
        except IndexError:
            try:
                frequency = kwds['frequency']
            except KeyError:
                try:
                    frequency = kwds['_frequency']
                except KeyError:
                    try:
                        frequency = args[0]
                        if not isinstance(frequency, int):
                            raise TypeError('frequency must must be integer. The correct SPI for SteppableBasePy is'
                                            ' e.g. SteppableBasePy(frequency=10)')
                    except IndexError:

                        frequency = 1

        SteppablePy.__init__(self)
        SBMLSolverHelper.__init__(self)

        # free floating SBML model accessor
        self.sbml = GlobalSBMLFetcher()

        # SBMLSolverHelper.__init__(self)
        self.frequency = frequency
        # self._simulator = simulator
        self.__modulesToUpdateDict = OrderedDict()

        # legacy API
        # self.addNewPlotWindow = self.add_new_plot_window
        # self.createScalarFieldPy = self.create_scalar_field_py
        # self.everyPixelWithSteps = self.every_pixel_with_steps
        # self.everyPixel = self.every_pixel
        # self.getCellNeighborDataList = self.get_cell_neighbor_data_list
        # self.attemptFetchingCellById = self.fetch_cell_by_id

        self.field = FieldFetcher()

        # plugin declarations - including legacy members
        self.neighbor_tracker_plugin = None
        self.neighborTrackerPlugin = None
        self.focal_point_plasticity_plugin = None
        self.focalPointPlasticityPlugin = None
        self.volume_tracker_plugin = None

        self.cell_field = None
        self.cellField = None

        self.plugin_init_dict = {
            "NeighborTracker": ['neighbor_tracker_plugin', 'neighborTrackerPlugin'],
            "FocalPointPlasticity": ['focal_point_plasticity_plugin', 'focalPointPlasticityPlugin'],
            "PixelTracker": ['pixel_tracker_plugin', 'pixelTrackerPlugin'],
            "AdhesionFlex": ['adhesion_flex_plugin', 'adhesionFlexPlugin'],
            "PolarizationVector": ['polarization_vector_plugin', 'polarizationVectorPlugin'],
            "Polarization23": ['polarization_23_plugin', 'polarization23Plugin'],
            "CellOrientation": ['cell_orientation_plugin', 'cellOrientationPlugin'],
            "ContactOrientation": ['contact_orientation_plugin', 'contactOrientationPlugin'],
            "ContactLocalProduct": ['contact_local_product_plugin', 'contactLocalProductPlugin'],
            "LengthConstraint": ['length_constraint_plugin', 'lengthConstraintPlugin'],
            "ConnectivityGlobal": ['connectivity_global_plugin', 'connectivityGlobalPlugin'],
            "ConnectivityLocalFlex": ['connectivity_local_flex_plugin', 'connectivityLocalFlexPlugin'],
            "Chemotaxis": ['chemotaxis_plugin', 'chemotaxisPlugin'],
        }

        # used by clone attributes functions
        self.clonable_attribute_names = ['lambdaVolume', 'targetVolume', 'targetSurface', 'lambdaSurface',
                                         'targetClusterSurface', 'lambdaClusterSurface', 'type', 'lambdaVecX',
                                         'lambdaVecY', 'lambdaVecZ', 'fluctAmpl']

    @property
    def simulator(self):
        return self._simulator()

    @simulator.setter
    def simulator(self, simulator):
        self._simulator = simulator

    @property
    def potts(self):
        return self.simulator.getPotts()

    # @property
    # def cellField(self):
    #     return self.potts.getCellFieldG()

    @property
    def dim(self):
        return self.cell_field.getDim()

    @property
    def inventory(self):
        return self.simulator.getPotts().getCellInventory()

    @property
    def clusterInventory(self) -> object:
        return self.inventory.getClusterInventory()

    def process_steering_panel_data(self):
        """
        Function to be implemented in steppable where we react to changes in the steering panel
        :return:
        """
        pass

    def add_steering_panel(self):
        """
        To be implemented in the subclass
        :return:
        """

    def process_steering_panel_data_wrapper(self):
        """
        Calls process_steering_panel_data if and only if there are dirty
        parameters in the steering panel model
        :return: None
        """
        if self.steering_param_dirty():
            self.process_steering_panel_data()

        # NOTE: resetting of the dirty flag for the steering
        # panel model is done in the SteppableRegistry's "step" function

    def add_steering_param(self, name, val, min_val=None, max_val=None, decimal_precision=3, enum=None,
                           widget_name=None):
        """
        Adds steering parameter
        :param name:
        :param val:
        :param min_val:
        :param max_val:
        :param decimal_precision:
        :param enum:
        :param widget_name:
        :return:
        """
        pg = CompuCellSetup.persistent_globals
        steering_param_dict = pg.steering_param_dict

        if self.mcs >= 0:
            raise RuntimeError(
                'Steering Parameters Can only be added in "__init__" or "start" function of the steppable')

        if name in steering_param_dict.keys():
            raise RuntimeError(
                'Steering parameter named {} has already been defined. Please use different parameter name'.format(
                    name))

        steering_param_dict[name] = SteeringParam(name=name, val=val, min_val=min_val, max_val=max_val,
                                                  decimal_precision=decimal_precision, enum=enum,
                                                  widget_name=widget_name)

    @staticmethod
    def get_steering_param(name: str) -> object:
        """
        Fetches value of the steering parameter
        :param name: parameter name
        :return: value
        """

        try:
            return CompuCellSetup.persistent_globals.steering_param_dict[name].val
        except KeyError:
            raise RuntimeError('Could not find steering_parameter named {}'.format(name))

    def steering_param_dirty(self, name=None):
        """
        Checks if a given steering parameter is dirty or if name is None if any of the parameters are dirty

        :param name:{str} name of the parameter
        True gets returned (False otherwise)
        :return:{bool} dirty flag
        """
        pg = CompuCellSetup.persistent_globals

        with pg.steering_panel_synchronizer:

            if name is not None:
                return self.get_steering_param(name=name).dirty_flag
            else:
                for p_name, steering_param in CompuCellSetup.persistent_globals.steering_param_dict.items():
                    if steering_param.dirty_flag:
                        return True
                return False

    def set_steering_param_dirty(self, name=None, flag=True):
        """
        Sets dirty flag for given steering parameter or if name is None all parameters
        have their dirty flag set to a given boolean value

        :param name:{str} name of the parameter
        :param flag:{bool} dirty_flag
        :return:None
        """
        pg = CompuCellSetup.persistent_globals
        with pg.steering_panel_synchronizer:
            if name is not None:
                self.get_steering_param(name=name).dirty_flag = flag
            else:
                for p_name, steering_param in CompuCellSetup.persistent_globals.steering_param_dict.items():
                    steering_param.dirty_flag = flag

    def fetch_loaded_plugins(self) -> None:
        """
        Processes self.plugin_init_dict and initializes member variables according to specification in
        self.plugin_init_dict. relies on fixed naming convention for plugin accessor functions defined in
        pyinterface/CompuCellPython/CompuCellExtraDeclarations.i in  PLUGINACCESSOR macro
        :return:
        """

        # Special handling of VolumeTrackerPlugin - used in cell field numpy-like array operations
        if self.simulator.pluginManager.isLoaded("VolumeTracker"):
            self.volume_tracker_plugin = CompuCell.getVolumeTrackerPlugin()
            # used in setitem function in SWIG CELLFIELDEXTEDER macro CompuCell.i
            self.cell_field.volumeTrackerPlugin = self.volume_tracker_plugin
            # self.potts.getCellFieldG().volumeTrackerPlugin =  self.volume_tracker_plugin

        for plugin_name, member_var_list in self.plugin_init_dict.items():
            if self.simulator.pluginManager.isLoaded(plugin_name):
                accessor_fcn_name = 'get' + plugin_name + 'Plugin'
                try:
                    accessor_function = getattr(CompuCell, accessor_fcn_name)
                except AttributeError:
                    warnings.warn('Could not locate {accessor_fcn_name} member of CompuCell python module')
                    for plugin_member_name in member_var_list:
                        setattr(self, plugin_member_name, None)

                    continue

                plugin_obj = accessor_function()

                for plugin_member_name in member_var_list:
                    setattr(self, plugin_member_name, plugin_obj)

            else:
                # in case the plugin is not loaded we initialize member variables associated with the plugin to None
                for plugin_member_name in member_var_list:
                    setattr(self, plugin_member_name, None)

    def core_init(self, reinitialize_cell_types=True):

        # self.potts = self.simulator.getPotts()
        self.cell_field = self.potts.getCellFieldG()
        self.cellField = self.cell_field
        # self.dim = self.cellField.getDim()
        # self.inventory = self.simulator.getPotts().getCellInventory()
        # self.clusterInventory = self.inventory.getClusterInventory()
        self.cell_list = CellList(self.inventory)
        self.cellList = self.cell_list
        self.cell_list_by_type = CellListByType(self.inventory)
        self.cellListByType = self.cell_list_by_type
        self.cluster_list = ClusterList(self.inventory)
        self.clusterList = self.cluster_list
        self.clusters = Clusters(self.inventory)
        self.mcs = -1

        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}

        persistent_globals = CompuCellSetup.persistent_globals
        persistent_globals.attach_dictionary_to_cells()

        type_id_type_name_dict = extract_type_names_and_ids()

        if reinitialize_cell_types:
            for type_id, type_name in type_id_type_name_dict.items():
                self.typename_to_attribute(cell_type_name=type_name, type_id=type_id)
                # setattr(self, type_name.upper(), type_id)

        self.fetch_loaded_plugins()

        return
        self.potts = self.simulator.getPotts()

        self.cell_field = self.potts.getCellFieldG()
        self.dim = self.cell_field.getDim()
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.clusterInventory = self.inventory.getClusterInventory()
        self.cell_list = CellList(self.inventory)
        self.cell_list_by_type = CellListByType(self.inventory)
        self.cluster_list = ClusterList(self.inventory)
        self.clusters = Clusters(self.inventory)
        self.mcs = -1

        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}

        persistent_globals = CompuCellSetup.persistent_globals
        persistent_globals.attach_dictionary_to_cells()

        type_id_type_name_dict = extract_type_names_and_ids()

        for type_id, type_name in type_id_type_name_dict.items():
            self.typename_to_attribute(cell_type_name=type_name, type_id=type_id)
            # setattr(self, type_name.upper(), type_id)

    def typename_to_attribute(self, cell_type_name: str, type_id: int) -> None:
        """
        sets steppable attribute based on type name
        Performs basic sanity checks
        :param cell_type_name:{str}
        :param type_id:{str}
        :return:
        """

        if cell_type_name.isspace() or not len(cell_type_name.strip()):
            raise AttributeError('cell type "{}" contains whitespaces'.format(cell_type_name))

        if not cell_type_name[0].isalpha():
            raise AttributeError('Invalid cell type "{}" . Type name must start with a letter'.format(cell_type_name))

        cell_type_name_attr = cell_type_name.upper()

        try:
            getattr(self, cell_type_name_attr)
            attribute_already_exists = True
        except AttributeError:
            attribute_already_exists = False

        if attribute_already_exists:
            raise AttributeError('Could not convert cell type {cell_type} to steppable attribute. '
                                 'Attribute {attr_name} already exists . Please change your cell type name'.format(
                cell_type=cell_type_name, attr_name=cell_type_name_attr
            ))

        setattr(self, cell_type_name_attr, type_id)

    def stop_simulation(self):
        """
        Stops simulation
        :return:
        """

        CompuCellSetup.stop_simulation()

    def init(self, _simulator):
        """

        :param _simulator:
        :return:
        """

    @deprecated(version='4.0.0', reason="You should use : add_new_plot_window")
    def addNewPlotWindow(self, _title, _xAxisTitle, _yAxisTitle, _xScaleType='linear', _yScaleType='linear', _grid=True,
                         _config_options=None):
        return self.add_new_plot_window(title=_title, x_axis_title=_xAxisTitle, y_axis_title=_yAxisTitle,
                                        x_scale_type=_xScaleType, y_scale_type=_yScaleType,
                                        grid=_grid, config_options=_config_options)

    def add_new_plot_window(self, title: str, x_axis_title: str, y_axis_title: str, x_scale_type: str = 'linear',
                            y_scale_type: str = 'linear', grid: bool = True,
                            config_options: object = None) -> object:

        if title in self.plot_dict.keys():
            raise RuntimeError('PLOT WINDOW: ' + title + ' already exists. Please choose a different name')

        pW = CompuCellSetup.simulation_player_utils.add_new_plot_window(title, x_axis_title, y_axis_title, x_scale_type,
                                                                        y_scale_type, grid,
                                                                        config_options=config_options)
        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}

        return pW

    @deprecated(version='4.0.0', reason="You should use : create_scalar_field_py")
    def createScalarFieldPy(self, _fieldName):
        return self.create_scalar_field_py(fieldName=_fieldName)

    def create_scalar_field_py(self, fieldName: str) -> ExtraFieldAdapter:
        """
        Created extra visualization field
        :param fieldName: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=fieldName,
                                                                         field_type=SCALAR_FIELD_NPY)

    @deprecated(version='4.0.0', reason="You should use : create_scalar_field_cell_level_py")
    def createScalarFieldCellLevelPy(self, _fieldName):
        return self.create_scalar_field_cell_level_py(field_name=_fieldName)

    def create_scalar_field_cell_level_py(self, field_name: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization field
        :param field_name: {str}
        :return:
        """
        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=field_name,
                                                                         field_type=SCALAR_FIELD_CELL_LEVEL)

    @deprecated(version='4.0.0', reason="You should use : create_vector_field_py")
    def createVectorFieldPy(self, _fieldName):
        return self.create_vector_field_py(field_name=_fieldName)

    def create_vector_field_py(self, field_name: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization vector field (voxel-based)
        :param field_name: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=field_name,
                                                                         field_type=VECTOR_FIELD_NPY)

    @deprecated(version='4.0.0', reason="You should use : create_vector_field_cell_level_py")
    def createVectorFieldCellLevelPy(self, _fieldName):
        return self.create_vector_field_cell_level_py(field_name=_fieldName)

    def create_vector_field_cell_level_py(self, field_name: str) -> ExtraFieldAdapter:
        """
        Creates extra visualization vector field (voxel-based)
        :param field_name: {str}
        :return:
        """

        return CompuCellSetup.simulation_player_utils.create_extra_field(field_name=field_name,
                                                                         field_type=VECTOR_FIELD_CELL_LEVEL)

    @deprecated(version='4.0.0', reason="You should use : every_pixel_with_steps")
    def everyPixelWithSteps(self, step_x, step_y, step_z):
        return self.every_pixel_with_steps(step_x=step_x, step_y=step_y, step_z=step_z)

    def every_pixel_with_steps(self, step_x, step_y, step_z):
        """
        Helper function called by every_pixel method. See documentation of every_pixel for details
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """
        for x in range(0, self.dim.x, step_x):
            for y in range(0, self.dim.y, step_y):
                for z in range(0, self.dim.z, step_z):
                    yield x, y, z

    @deprecated(version='4.0.0', reason="You should use : every_pixel")
    def everyPixel(self, step_x=1, step_y=1, step_z=1):
        return self.every_pixel(step_x=step_x, step_y=step_y, step_z=step_z)

    def every_pixel(self, step_x=1, step_y=1, step_z=1):
        """
        Returns iterator that walks through pixels of the lattixe. Step variables
        determine if we walk through every pixel - step=1 in this case
        or if we jump step variables then are > 1
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """

        if step_x == 1 and step_y == 1 and step_z == 1:

            return itertools.product(range(self.dim.x), range(self.dim.y), range(self.dim.z))
        else:
            return self.every_pixel_with_steps(step_x, step_y, step_z)

    def get_xml_element(self, tag: str) -> Union[XMLElemAdapter, None]:
        """
        Fetches XML element by id. Returns XMLElementAdapter object that provides natural way of manipulating
        properties of the underlying XML element

        :param tag: {str}  xml element identifier - must be present in the xml
        :return: {XMLElemAdapter} xml element adapter
        """
        xml_id_locator = CompuCellSetup.persistent_globals.xml_id_locator

        if xml_id_locator is None:
            return None

        element_adapter = xml_id_locator.get_xml_element(tag=tag)

        return element_adapter

    @deprecated(version='4.0.0', reason="You should use : get_cell_neighbor_data_list")
    def getCellNeighborDataList(self, _cell):
        return self.get_cell_neighbor_data_list(cell=_cell)

    def get_cell_neighbor_data_list(self, cell):
        if not self.neighbor_tracker_plugin:
            raise AttributeError('Could not find NeighborTrackerPlugin')

        return CellNeighborListFlex(self.neighbor_tracker_plugin, cell)

    @deprecated(version='4.0.0', reason="You should use : fetch_cell_by_id")
    def attemptFetchingCellById(self, _id):
        return self.fetch_cell_by_id(cell_id=_id)

    def fetch_cell_by_id(self, cell_id: int) -> Union[None, object]:
        """
        Fetches cell by id. If cell does not exist it returns None
        :param cell_id: cell id
        :return: successfully fetched cell id or None
        """
        return self.inventory.attemptFetchingCellById(cell_id)

    def getFocalPointPlasticityDataList(self, _cell):
        if self.focal_point_plasticity_plugin:
            return FocalPointPlasticityDataList(self.focal_point_plasticity_plugin, _cell)

        return None

    def getInternalFocalPointPlasticityDataList(self, _cell):
        if self.focal_point_plasticity_plugin:
            return InternalFocalPointPlasticityDataList(self.focal_point_plasticity_plugin, _cell)

        return None

    def getAnchorFocalPointPlasticityDataList(self, _cell):
        if self.focal_point_plasticity_plugin:
            return AnchorFocalPointPlasticityDataList(self.focal_point_plasticity_plugin, _cell)

        return None

    @deprecated(version='4.0.0', reason="You should use : build_wall")
    def buildWall(self, type):
        return self.build_wall(cell_type=type)

    def build_wall(self, cell_type):
        # medium:
        if cell_type == 0:
            cell = CompuCell.getMediumCell()
        else:
            cell = self.potts.createCell()
            cell.type = cell_type

        index_of1 = -1
        dim_local = [self.dim.x, self.dim.y, self.dim.z]

        for idx in range(len(dim_local)):

            if dim_local[idx] == 1:
                index_of1 = idx
                break

        # this could be recoded in a more general way
        # 2D case
        if index_of1 >= 0:

            if index_of1 == 2:
                # xy plane simulation
                self.cell_field[0:self.dim.x, 0, 0] = cell
                self.cell_field[0:self.dim.x, self.dim.y - 1:self.dim.y, 0] = cell
                self.cell_field[0, 0:self.dim.y, 0] = cell
                self.cell_field[self.dim.x - 1:self.dim.x, 0:self.dim.y, 0:0] = cell

            elif index_of1 == 0:
                # yz simulation
                self.cell_field[0, 0:self.dim.y, 0] = cell
                self.cell_field[0, 0:self.dim.y, self.dim.z - 1:self.dim.z] = cell
                self.cell_field[0, 0, 0:self.dim.z] = cell
                self.cell_field[0, self.dim.y - 1:self.dim.y, 0:self.dim.z] = cell

            elif index_of1 == 1:
                # xz simulation
                self.cell_field[0:self.dim.x, 0, 0] = cell
                self.cell_field[0:self.dim.x, 0, self.dim.z - 1:self.dim.z] = cell
                self.cell_field[0, 0, 0:self.dim.z] = cell
                self.cell_field[self.dim.x - 1:self.dim.x, 0, 0:self.dim.z] = cell
        else:
            # 3D case
            # wall 1 (front)
            self.cell_field[0:self.dim.x, 0:self.dim.y, 0] = cell
            # wall 2 (rear)
            self.cell_field[0:self.dim.x, 0:self.dim.y, self.dim.z - 1] = cell
            # wall 3 (bottom)
            self.cell_field[0:self.dim.x, 0, 0:self.dim.z] = cell
            # wall 4 (top)
            self.cell_field[0:self.dim.x, self.dim.y - 1, 0:self.dim.z] = cell
            # wall 5 (left)
            self.cell_field[0, 0:self.dim.y, 0:self.dim.z] = cell
            # wall 6 (right)
            self.cell_field[self.dim.x - 1, 0:self.dim.y, 0:self.dim.z] = cell

    @deprecated(version='4.0.0', reason="You should use : destroy_wall")
    def destroyWall(self):
        return self.destroy_wall()

    def destroy_wall(self):
        # build wall of Medium
        self.build_wall(0)

    @deprecated(version='4.0.0', reason="You should use : resize_and_shift_lattice")
    def resizeAndShiftLattice(self, _newSize, _shiftVec=(0, 0, 0)):
        return self.resize_and_shift_lattice(new_size=_newSize, shift_vec=_shiftVec)

    def resize_and_shift_lattice(self, new_size, shift_vec=(0, 0, 0)):
        """
        resizes and shits lattice. Checks if the operation is possible , if not the action is abandoned

        :param new_size: {list} new size
        :param shift_vec: {list} shift vector
        :return: None
        """

        if self.potts.getBoundaryXName().lower() == 'periodic' \
                or self.potts.getBoundaryYName().lower() == 'periodic' \
                or self.potts.getBoundaryZName().lower() == 'periodic':
            raise EnvironmentError('Cannot resize lattice with Periodic Boundary Conditions')

        # converting new size to integers
        new_size = list(map(int, new_size))
        # converting shift vec to integers
        shift_vec = list(map(int, shift_vec))

        ok_flag = self.volume_tracker_plugin.checkIfOKToResize(CompuCell.Dim3D(new_size[0], new_size[1], new_size[2]),
                                                               CompuCell.Dim3D(shift_vec[0], shift_vec[1],
                                                                               shift_vec[2]))
        print('ok_flag=', ok_flag)
        if not ok_flag:
            warnings.warn('WARNING: Lattice Resize Denied. '
                          'The proposed lattice resizing/shift would lead to disappearance of cells.', Warning)
            return

        old_geometry_dimensionality = 2
        if self.dim.x > 1 and self.dim.y > 1 and self.dim.z > 1:
            old_geometry_dimensionality = 3

        new_geometry_dimensionality = 2
        if new_size[0] > 1 and new_size[1] > 1 and new_size[2] > 1:
            new_geometry_dimensionality = 3

        if new_geometry_dimensionality != old_geometry_dimensionality:
            raise RuntimeError('Changing dimmensionality of simulation from 2D to 3D is not supported. '
                               'It also makes little sense as 2D and 3D simulations have different mathematical properties. '
                               'Please see CPM literature for more details.')

        self.potts.resizeCellField(CompuCell.Dim3D(new_size[0], new_size[1], new_size[2]),
                                   CompuCell.Dim3D(shift_vec[0], shift_vec[1], shift_vec[2]))
        #         if sum(shift_vec)==0: # there is no shift in cell field
        #             return

        # posting CC3DEventLatticeResize so that participating modules can react
        resize_event = CompuCell.CC3DEventLatticeResize()
        resize_event.oldDim = self.dim
        resize_event.newDim = CompuCell.Dim3D(new_size[0], new_size[1], new_size[2])
        resize_event.shiftVec = CompuCell.Dim3D(shift_vec[0], shift_vec[1], shift_vec[2])

        self.simulator.postEvent(resize_event)

        self.__init__(self.frequency)
        self.core_init(reinitialize_cell_types=False)

        # with new cell field and possibly other fields  we have to reinitialize steppables
        for steppable in CompuCellSetup.persistent_globals.steppable_registry.allSteppables():
            if steppable != self:
                steppable.__init__(steppable.frequency)

    @deprecated(version='4.0.0', reason="You should use : distance_vector")
    def distanceVector(self, _from, _to):
        return self.distance_vector(p1=_from, p2=_to)

    def distance_vector(self, p1, p2):
        """
        This function will calculate distance vector between  two points - (_to-_from)
        This is most straightforward implementation and will ignore periodic boundary conditions if such are present
        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        return np.array([float(p2[0] - p1[0]), float(p2[1] - p1[1]), float(p2[2] - p1[2])])

    def distance(self, p1, p2):
        """
        Distance between two points. Assumes non-periodic boundary conditions
        :return: {float} "naive" distance between two points
        """
        return self.vectorNorm(self.distance_vector(p1, p2))

    @deprecated(version='4.0.0', reason="You should use : invariant_distance")
    def invariantDistance(self, _from, _to):
        return self.invariant_distance(p1=_from, p2=_to)

    def invariant_distance(self, p1, p2):
        """
        Distance between two points. Assumes periodic boundary conditions
        - or simply makes sure that no component of distance vector
        is greater than 1/2 corresponding dimension
        :return: {float} invariant distance between two points
        """

        return self.vector_norm(self.invariant_distance_vector(p1, p2))

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector_integer")
    def invariantDistanceVectorInteger(self, _from, _to):
        return self.invariant_distance_vector_integer(p1=_from, p2=_to)

    def invariant_distance_vector_integer(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Point3D points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place

        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        dist_vec = CompuCell.distanceVectorInvariant(p2, p1, self.dim)
        return np.array([float(dist_vec.x), float(dist_vec.y), float(dist_vec.z)])

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector")
    def invariantDistanceVector(self, _from, _to):
        return self.invariant_distance_vector(p1=_from, p2=_to)

    def invariant_distance_vector(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Coordinates3D<double> points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place
        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        dist_vec = CompuCell.distanceVectorCoordinatesInvariant(p2, p1, self.dim)
        return np.array([dist_vec.x, dist_vec.y, dist_vec.z])

    @deprecated(version='4.0.0', reason="You should use : vectorNorm")
    def vectorNorm(self, _vec):
        return self.vector_norm(vec=_vec)

    @staticmethod
    def vector_norm(vec):
        """
        Computes norm of a vector
        :param vec: vector
        :return:
        """

        return np.linalg.norm(vec)

    @deprecated(version='4.0.0', reason="You should use : distance_vector_between_cells")
    def distanceVectorBetweenCells(self, _cell_from, _cell_to):
        return self.distance_vector_between_cells(cell1=_cell_from, cell2=_cell_to)

    def distance_vector_between_cells(self, cell1, cell2):
        """
        This function will calculate distance vector between  COM's of cells  assuming non-periodic boundary conditions
        :return: {ndarray} distance vector
        """
        return self.distance_vector([cell1.xCOM, cell1.yCOM, cell1.zCOM], [cell2.xCOM, cell2.yCOM, cell2.zCOM])

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector_between_cells")
    def invariantDistanceVectorBetweenCells(self, _cell_from, _cell_to):
        return self.invariant_distance_vector_between_cells(cell1=_cell_from, cell2=_cell_to)

    def invariant_distance_vector_between_cells(self, cell1, cell2):
        """
        This function will calculate distance vector between  COM's of cells  assuming periodic boundary conditions
        - or simply makes sure that no component of distance vector
        is greater than 1/2 corresponding dimension
        :return: {ndarray} distance vector
        """
        return self.invariant_distance_vector([cell1.xCOM, cell1.yCOM, cell1.zCOM],
                                              [cell2.xCOM, cell2.yCOM, cell2.zCOM])

    @deprecated(version='4.0.0', reason="You should use : distance_between_cells")
    def distanceBetweenCells(self, _cell_from, _cell_to):
        return self.distance_between_cells(cell1=_cell_from, cell2=_cell_to)

    def distance_between_cells(self, cell1, cell2):
        """
        Distance between COM's between cells. Assumes non-periodic boundary conditions
        :return: naive distance between COM of cells
        """

        return self.vector_norm(self.distance_vector_between_cells(cell1, cell2))

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_between_cells")
    def invariantDistanceBetweenCells(self, _cell_from, _cell_to):
        return self.invariant_distance_between_cells(cell1=_cell_from, cell2=_cell_to)

    def invariant_distance_between_cells(self, cell1, cell2):
        """
        Distance between COM's of two cells. Assumes periodic boundary conditions
        - or simply makes sure that no component of distance vector
        is greater than 1/2 corresponding dimension
        :return: invariant distance between COM of cells
        """
        return self.vector_norm(self.invariant_distance_vector_between_cells(cell1, cell2))

    @deprecated(version='4.0.0', reason="You should use : new_cell")
    def newCell(self, type=0):
        return self.new_cell(cell_type=type)

    def new_cell(self, cell_type=0):
        cell = self.potts.createCell()
        cell.type = cell_type
        return cell

    @deprecated(version='4.0.0', reason="You should use : get_pixel_neighbors_based_on_neighbor_order")
    def getPixelNeighborsBasedOnNeighborOrder(self, _pixel, _neighborOrder=1):
        return self.get_pixel_neighbors_based_on_neighbor_order(pixel=_pixel, neighbor_order=_neighborOrder)

    def get_pixel_neighbors_based_on_neighbor_order(self, pixel, neighbor_order=1):
        """
        generator that returns a sequence of pixel neighbors up to specified neighbor order
        :param pixel: {CompuCell.Point3D} pixel
        :param neighbor_order: {int} neighbor order
        :return:
        """

        boundary_strategy = CompuCell.BoundaryStrategy.getInstance()
        max_neighbor_index = boundary_strategy.getMaxNeighborIndexFromNeighborOrder(neighbor_order)

        # returning a list might be safer in some situations For now sticking with the generator
        for i in range(max_neighbor_index + 1):
            pixel_neighbor = boundary_strategy.getNeighborDirect(pixel, i)
            if pixel_neighbor.distance:
                # neighbor is valid
                yield pixel_neighbor

    @deprecated(version='4.0.0', reason="You should use : get_cell_pixel_list")
    def getCellPixelList(self, _cell):
        return self.get_cell_pixel_list(cell=_cell)

    def get_cell_pixel_list(self, cell):
        """
        For a given cell returns list of cell pixels

        :param cell: {CompuCell.CellG}
        :return: {list} list of cell pixels
        """
        if self.pixelTrackerPlugin:
            return CellPixelList(self.pixelTrackerPlugin, cell)

        return None

    @deprecated(version='4.0.0', reason="You should use : move_cell")
    def moveCell(self, cell, shiftVector):
        return self.move_cell(cell=cell, shift_vector=shiftVector)

    def move_cell(self, cell, shift_vector):
        """
        Moves cell by shift_vector
        :param cell: {CompuCell.CellG} cell
        :param shift_vector: {tuple,  list, array, or CompuCell.Point3D} 3-element list specifying shoft vector
        :return: None
        """

        # we have to make two list of pixels :
        # used to hold pixels to delete
        pixels_to_delete = []
        # used to hold pixels to move
        pixels_to_move = []

        shift_vec = CompuCell.Point3D()
        if isinstance(shift_vector, list) or isinstance(shift_vector, tuple):
            shift_vec.x = shift_vector[0]
            shift_vec.y = shift_vector[1]
            shift_vec.z = shift_vector[2]
        else:
            shift_vec = shift_vector
        # If we try to reassign pixels in the loop where we iterate over pixel data
        # we will corrupt the container so in the loop below all we will do is to populate the two list mentioned above
        pixel_list = self.get_cell_pixel_list(cell)
        pt = CompuCell.Point3D()

        for pixelTrackerData in pixel_list:
            pt.x = pixelTrackerData.pixel.x + shift_vec.x
            pt.y = pixelTrackerData.pixel.y + shift_vec.y
            pt.z = pixelTrackerData.pixel.z + shift_vec.z
            # here we are making a copy of the cell
            pixels_to_delete.append(CompuCell.Point3D(pixelTrackerData.pixel))

            if self.check_if_in_the_lattice(pt):
                pixels_to_move.append(CompuCell.Point3D(pt))
                # self.cellField.set(pt,cell)

        # Now we will move cell
        for pixel in pixels_to_move:
            self.cell_field[pixel.x, pixel.y, pixel.z] = cell

        # Now we will delete old pixels
        medium_cell = CompuCell.getMediumCell()
        for pixel in pixels_to_delete:
            self.cell_field[pixel.x, pixel.y, pixel.z] = medium_cell

    @deprecated(version='4.0.0', reason="You should use : check_if_in_the_lattice")
    def checkIfInTheLattice(self, _pt):
        return self.check_if_in_the_lattice(pt=_pt)

    def check_if_in_the_lattice(self, pt):
        if pt.x >= 0 and pt.x < self.dim.x and pt.y >= 0 and pt.y < self.dim.y and pt.z >= 0 and pt.z < self.dim.z:
            return True
        return False

    # def getCopyOfCellPixels(self, _cell, _format=CC3D_FORMAT):

    def get_copy_of_cell_pixels(self, cell, format=CC3D_FORMAT):

        try:
            if format == SteppableBasePy.CC3D_FORMAT:
                return [CompuCell.Point3D(pixelTrackerData.pixel) for pixelTrackerData in
                        self.get_cell_pixel_list(cell)]
            else:
                return [(pixelTrackerData.pixel.x, pixelTrackerData.pixel.y, pixelTrackerData.pixel.z) for
                        pixelTrackerData in self.get_cell_pixel_list(cell)]
        except:
            raise AttributeError('Could not find PixelTracker Plugin')

    @deprecated(version='4.0.0', reason="You should use : delete_cell")
    def deleteCell(self, cell):
        return self.delete_cell(cell=cell)

    def delete_cell(self, cell):
        """
        Deletes given cell by overwriting its pixels with medium pixels
        :param cell:
        :return:
        """
        # returns list of tuples
        pixels_to_delete = self.get_copy_of_cell_pixels(cell, SteppableBasePy.TUPLE_FORMAT)
        medium_cell = CompuCell.getMediumCell()
        for pixel in pixels_to_delete:
            self.cell_field[pixel[0], pixel[1], pixel[2]] = medium_cell

    def cloneAttributes(self, sourceCell, targetCell, no_clone_key_dict_list=[]):
        return self.clone_attributes(source_cell=sourceCell, target_cell=targetCell,
                                     no_clone_key_dict_list=no_clone_key_dict_list)

    def clone_attributes(self, source_cell, target_cell, no_clone_key_dict_list=[]):
        """
        Copies attributes from source cell to target cell. Users can specify which attributes should not be clones
        using no_clone_key_dict_list
        :param source_cell: {CompuCell.CellG} source cell
        :param target_cell: {CompuCell.CellG} target cell
        :param no_clone_key_dict_list: {list} list of dictionaries of attributes that are not to be cloned
        :return:
        """

        # clone "C++" attributes
        for attrName in self.clonable_attribute_names:
            setattr(target_cell, attrName, getattr(source_cell, attrName))

        # clone dictionary
        for key, val in source_cell.dict.items():

            if key in no_clone_key_dict_list:
                continue

            elif key == 'SBMLSolver':
                self.copy_sbml_simulators(from_cell=source_cell, to_cell=target_cell)
            else:
                # copying the rest of dictionary entries
                target_cell.dict[key] = deepcopy(source_cell.dict[key])

        # now copy data associated with plugins
        # AdhesionFlex
        if self.adhesionFlexPlugin:
            source_adhesion_vector = self.adhesionFlexPlugin.getAdhesionMoleculeDensityVector(source_cell)
            self.adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(target_cell, source_adhesion_vector)

        # PolarizationVector
        if self.polarizationVectorPlugin:
            source_polarization_vector = self.polarizationVectorPlugin.getPolarizationVector(source_cell)
            self.polarizationVectorPlugin.setPolarizationVector(target_cell, source_polarization_vector[0],
                                                                source_polarization_vector[1],
                                                                source_polarization_vector[2])

        # polarization23Plugin
        if self.polarization23Plugin:
            pol_vec = self.polarization23Plugin.getPolarizationVector(source_cell)
            self.polarization23Plugin.setPolarizationVector(target_cell, pol_vec)
            pol_mark = self.polarization23Plugin.getPolarizationMarkers(source_cell)
            self.polarization23Plugin.setPolarizationMarkers(target_cell, pol_mark[0], pol_mark[1])
            lam = self.polarization23Plugin.getLambdaPolarization(source_cell)
            self.polarization23Plugin.setLambdaPolarization(target_cell, lam)

        # CellOrientationPlugin
        if self.cellOrientationPlugin:
            lam = self.cellOrientationPlugin.getLambdaCellOrientation(source_cell)
            self.cellOrientationPlugin.setLambdaCellOrientation(target_cell, lam)

        # ContactOrientationPlugin
        if self.contactOrientationPlugin:
            o_vec = self.contactOrientationPlugin.getOriantationVector(source_cell)
            self.contactOrientationPlugin.setOriantationVector(target_cell, o_vec.x, o_vec.y, o_vec.z)
            self.contactOrientationPlugin.setAlpha(target_cell, self.contactOrientationPlugin.getAlpha(source_cell))

        # ContactLocalProductPlugin
        if self.contactLocalProductPlugin:
            c_vec = self.contactLocalProductPlugin.getCadherinConcentrationVec(source_cell)
            self.contactLocalProductPlugin.setCadherinConcentrationVec(target_cell, c_vec)

        # LengthConstraintPlugin
        if self.lengthConstraintPlugin:
            lam = self.lengthConstraintPlugin.getLambdaLength(source_cell)
            tl = self.lengthConstraintPlugin.getTargetLength(source_cell)
            mtl = self.lengthConstraintPlugin.getMinorTargetLength(source_cell)
            self.lengthConstraintPlugin.setLengthConstraintData(target_cell, lam, tl, mtl)

        # ConnectivityGlobalPlugin
        if self.connectivityGlobalPlugin:
            cs = self.connectivityGlobalPlugin.getConnectivityStrength(source_cell)
            self.connectivityGlobalPlugin.setConnectivityStrength(target_cell, cs)

        # ConnectivityLocalFlexPlugin
        if self.connectivityLocalFlexPlugin:
            cs = self.connectivityLocalFlexPlugin.getConnectivityStrength(source_cell)
            self.connectivityLocalFlexPlugin.setConnectivityStrength(target_cell, cs)

        # Chemotaxis
        if self.chemotaxisPlugin:
            field_names = self.chemotaxisPlugin.getFieldNamesWithChemotaxisData(source_cell)

            for fieldName in field_names:
                source_chd = self.chemotaxisPlugin.getChemotaxisData(source_cell, fieldName)
                target_chd = self.chemotaxisPlugin.addChemotaxisData(target_cell, fieldName)

                target_chd.setLambda(source_chd.getLambda())
                target_chd.saturationCoef = source_chd.saturationCoef
                target_chd.setChemotaxisFormulaByName(source_chd.formulaName)
                target_chd.assignChemotactTowardsVectorTypes(source_chd.getChemotactTowardsVectorTypes())

                # FocalPointPLasticityPlugin - this plugin has to be handled manually -
                # there is no good way to figure out which links shuold be copied from parent to daughter cell

    # def registerXMLElementUpdate(self, *args):
    #     '''this function registers core module XML Element from wchich XML subelement has been fetched.It returns XML subelement
    #     '''
    #     # element,coreElement=None,None
    #     # info=sys.version_info
    #     # if info[0]>=2 and info[1]>5:
    #     #     element,coreElement=self.getXMLElementAndModuleRoot(*args,returnModuleRoot=True)  # does not work in python 2.5 - syntax error
    #     # else:
    #     element, coreElement = self.getXMLElementAndModuleRoot(args, returnModuleRoot=True)
    #
    #     coreNameComposite = coreElement.getName()
    #     if coreElement.findAttribute('Name'):
    #         coreNameComposite += coreElement.getAttribute('Name')
    #     elif coreElement.findAttribute('Type'):
    #         coreNameComposite += coreElement.getAttribute('Type')
    #
    #     if element:
    #
    #         # now will register which modules were modified we will use this information when we call update function
    #         currentMCS = self.simulator.getStep()
    #         try:
    #             moduleDict = self.__modulesToUpdateDict[currentMCS]
    #             try:
    #                 moduleDict[coreNameComposite]
    #             except LookupError:
    #                 moduleDict['NumberOfModules'] += 1
    #                 moduleDict[coreNameComposite] = [coreElement, moduleDict['NumberOfModules']]
    #                 # # # print 'moduleDict[NumberOfModules]=',moduleDict['NumberOfModules']
    #
    #         except LookupError:
    #             self.__modulesToUpdateDict[currentMCS] = {coreNameComposite: [coreElement, 0], 'NumberOfModules': 0}
    #
    #     return element
    #
    # def getXMLAttributeValue(self, attr, *args):
    #     element = self.getXMLElement(*args)
    #     if element is not None:
    #         if element.findAttribute(attr):
    #             return element.getAttribute(attr)
    #         else:
    #             raise LookupError('Could not find attribute ' + attr + ' in ' + args)
    #     else:
    #         return None
    #
    # def setXMLAttributeValue(self, attr, value, *args):
    #     element = self.registerXMLElementUpdate(*args)
    #     if element:
    #         if element.findAttribute(attr):
    #             element.updateElementAttributes(d2mss({attr: value}))
    #
    # def updateXML(self):
    #     currentMCS = self.simulator.getStep()
    #     try:
    #         # trying to get dictionary of  modules for which XML has been modified during current step
    #         moduleDict = self.__modulesToUpdateDict[currentMCS]
    #     except LookupError:
    #         # if such dictionary does not exist we clean self.__modulesToUpdateDict deleteing whatever was stored before
    #         self.__modulesToUpdateDict = {}
    #         return
    #
    #     try:
    #         numberOfModules = moduleDict['NumberOfModules']
    #         del moduleDict['NumberOfModules']
    #     except LookupError:
    #         pass
    #
    #     # [1][1] refers to number denoting the order in which module was added
    #     # [1][1] refers to added element with order number being [1][1]
    #     list_of_tuples = sorted(moduleDict.items(), key=lambda x: x[1][1])
    #
    #     # # # print 'list_of_tuples=',list_of_tuples
    #     for elem_tuple in list_of_tuples:
    #         self.simulator.updateCC3DModule(elem_tuple[1][0])
    #
    # def getXMLElement(self, *args):
    #     element = None
    #
    #     if not len(args):
    #         return None
    #
    #     if type(args[0]) is not list:  # it is CC3DXMLElement
    #         element = args[0]
    #     else:
    #         element, moduleRoot = self.getXMLElementAndModuleRoot(*args)
    #
    #     return element if element else None
    #
    # def getXMLElementValue(self, *args):
    #
    #     element = self.getXMLElement(*args)
    #     return element.getText() if element else None
    #
    # def getXMLElementAndModuleRoot(self, *args, **kwds):
    #     ''' This fcn fetches xml element value and returns it as text. Potts, Plugin and steppable are special names and roots of these elements are fetched using simulator
    #         The implementation of this plugin may be simplified. Current implementation is least invasive and requires no changes apart from modifying PySteppables.
    #         This Function greatly simplifies access to XML data - one line  easily replaces  many lines of code
    #     '''
    #
    #     # depending on Python version we might need to pass "extra-tupple-wrapped"
    #     # positional arguments especially in situation when variable list arguments
    #     # are mixed with keyword arguments during function call
    #     if isinstance(args[0], tuple):
    #         args = args[0]
    #
    #     if not isinstance(args[0], list):  # it is CC3DXMLElement
    #         return args[0]
    #
    #     coreModuleElement = None
    #     tmpElement = None
    #     for arg in args:
    #         if type(arg) is list:
    #             if arg[0] == 'Potts':
    #                 coreModuleElement = self.simulator.getCC3DModuleData('Potts')
    #                 tmpElement = coreModuleElement
    #             elif arg[0] == 'Plugin':
    #                 counter = 0
    #                 for attrName in arg:
    #                     if attrName == 'Name':
    #                         pluginName = arg[counter + 1]
    #                         coreModuleElement = self.simulator.getCC3DModuleData('Plugin', pluginName)
    #                         tmpElement = coreModuleElement
    #                         break
    #                     counter += 1
    #
    #             elif arg[0] == 'Steppable':
    #                 counter = 0
    #                 for attrName in arg:
    #                     if attrName == 'Type':
    #                         steppableName = arg[counter + 1]
    #                         coreModuleElement = self.simulator.getCC3DModuleData('Steppable', steppableName)
    #                         tmpElement = coreModuleElement
    #                         break
    #                     counter += 1
    #             else:
    #                 # print 'XML FETCH=',arg
    #                 attrDict = None
    #                 if len(arg) >= 3:
    #                     attrDict = {}
    #                     for attr_tuple in zip(arg[1::2], arg[2::2]):
    #                         if attr_tuple[0] in attrDict.keys():
    #                             raise LookupError('Duplicate attribute name in the access path ' + str(args))
    #                         else:
    #                             attrDict[attr_tuple[0]] = attr_tuple[1]
    #                     attrDict = d2mss(attrDict)
    #                     # attrDict=d2mss(dict((tuple[0],tuple[1]) for tuple in izip(arg[1::2],arg[2::2])))
    #
    #                 if coreModuleElement is not None:
    #                     elemName = arg[0]
    #                     tmpElement = tmpElement.getFirstElement(arg[0],
    #                                                             attrDict) if attrDict is not None else tmpElement.getFirstElement(
    #                         arg[0])
    #
    #     if tmpElement is None:
    #         raise LookupError('Could not find element With the following access path', args)
    #
    #     if 'returnModuleRoot' in kwds.keys():
    #         return tmpElement, coreModuleElement
    #
    #     return tmpElement, None


class MitosisSteppableBase(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.mitosisSteppable = None
        self.parentCell = None
        self.childCell = None
        self._parent_child_position_flag = None

    def core_init(self, reinitialize_cell_types=True):
        SteppableBasePy.core_init(self, reinitialize_cell_types=reinitialize_cell_types)
        self.mitosisSteppable = CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parentCell = self.mitosisSteppable.parentCell
        self.childCell = self.mitosisSteppable.childCell

        # delayed initialization
        if self._parent_child_position_flag is not None:
            self.set_parent_child_position_flag(flag=self._parent_child_position_flag)

    @deprecated(version='4.0.0', reason="You should use : set_parent_child_position_flag")
    def setParentChildPositionFlag(self, _flag):
        return self.set_parent_child_position_flag(flag=_flag)

    def set_parent_child_position_flag(self, flag):
        """
        Specifies which position of the "child" cell after mitosis process
        :param flag:{int} 0 - parent child position will be randomized between mitosis event
        negative integer - parent appears on the 'left' of the child
        positive integer - parent appears on the 'right' of the child
        :return: None
        """
        if self.mitosisSteppable is not None:
            self.mitosisSteppable.setParentChildPositionFlag(int(flag))
        else:
            self._parent_child_position_flag = flag

    @deprecated(version='4.0.0', reason="You should use : get_parent_child_position_flag")
    def getParentChildPositionFlag(self, _flag):
        return self.get_parent_child_position_flag()

    def get_parent_child_position_flag(self):
        return self.mitosisSteppable.getParentChildPositionFlag()

    @deprecated(version='4.0.0', reason="You should use : clone_parent_2_child")
    def cloneParent2Child(self):
        return self.clone_parent_2_child()

    def clone_parent_2_child(self):
        """
        clones attributes of parent cell to daughter cell
        :return: None
        """
        # these calls seem to be necessary to ensure whatever is setin in mitosisSteppable (C++) is reflected in Python
        # self.parentCell=self.mitosisSteppable.parentCell
        # self.childCell=self.mitosisSteppable.childCell

        self.clone_attributes(source_cell=self.parentCell, target_cell=self.childCell, no_clone_key_dict_list=[])

    def update_attributes(self):
        """
        This function is supposed to be reimplemented in the subclass. It is called immediately after cell division
        takes place
        :return:
        """

        self.childCell.targetVolume = self.parentCell.targetVolume
        self.childCell.lambdaVolume = self.parentCell.lambdaVolume
        self.childCell.type = self.parentCell.type

    @deprecated(version='4.0.0', reason="You should use : init_parent_and_child_cells")
    def initParentAndChildCells(self):
        return self.init_parent_and_child_cells()

    def init_parent_and_child_cells(self):
        """
        Initializes self.parentCell and self.childCell to point to respective cell objects after mitosis
         is completed succesfully
        """

        self.parentCell = self.mitosisSteppable.parentCell
        self.childCell = self.mitosisSteppable.childCell

    def handle_mitosis_update_attributes(self, mitosis_done):
        """
        Performs actions and bookipping that has to be done after actual cell division happened.
        One of such actions is calling update_attributes function that users typically overload
        :param mitosis_done: {bool} flag indicating if mitosis has been sucessful
        :return: None
        """

        if mitosis_done:
            self.init_parent_and_child_cells()
            legacy_update_attributes_fcn = getattr(self, 'updateAttributes')

            if legacy_update_attributes_fcn is not None:
                warnings.warn('"updateAttribute function" is deprecated since 4.0.0. '
                              'Please use "update_attributes" in your'
                              ' mitosis subclass', DeprecationWarning)
                legacy_update_attributes_fcn()
            else:
                self.update_attributes()

    @deprecated(version='4.0.0', reason="You should use : divide_cell_random_orientation")
    def divideCellRandomOrientation(self, _cell):
        return self.divide_cell_random_orientation(cell=_cell)

    def divide_cell_random_orientation(self, cell):
        """
        Divides cell into two daughter cells along randomly chosen cleavage plane.
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a cell object that existed before division

        :param cell: {CompuCell.CellG} cell to divide
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisRandomOrientation(cell)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        # if self.mitosis_done:
        #     self.init_parent_and_child_cells()
        #
        #     self.update_attributes()
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cell_orientation_vector_based")
    def divideCellOrientationVectorBased(self, _cell, _nx, _ny, _nz):
        return self.divide_cell_orientation_vector_based(cell=_cell, nx=_nx, ny=_ny, nz=_nz)

    def divide_cell_orientation_vector_based(self, cell, nx, ny, nz):
        """
        Divides cell into two daughter cells along cleavage plane specified by a normal vector (nx, ny, nz).
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a cell object that existed before division
        :param cell: {CompuCell.CellG} cell to divide
        :param nx: {float} 'x' component of vector normal to the cleavage plane
        :param ny: {float} 'y' component of vector normal to the cleavage plane
        :param nz: {float} 'z' component of vector normal to the cleavage plane
        :return:
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisOrientationVectorBased(cell, nx, ny, nz)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        # if self.mitosis_done:
        #     self.init_parent_and_child_cells()
        #     self.update_attributes()
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cell_along_major_axis")
    def divideCellAlongMajorAxis(self, _cell):
        return self.divide_cell_along_major_axis(cell=_cell)

    def divide_cell_along_major_axis(self, cell):
        """
        Divides cell into two daughter cells along cleavage plane parallel to major axis of the cell.
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a cell object that existed before division
        :param cell: {CompuCell.CellG} cell to divide
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisAlongMajorAxis(cell)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        # if self.mitosis_done:
        #     self.init_parent_and_child_cells()
        #     legacy_update_attributes_fcn = getattr(self,'updateAttributes')
        #
        #     if legacy_update_attributes_fcn is not None:
        #         warnings.warn('"updateAttribute function" is deprecated since 4.0.0. '
        #                       'Please use "update_attributes" in your'
        #                       ' mitosis subclass', DeprecationWarning)
        #         legacy_update_attributes_fcn()
        #     else:
        #         self.update_attributes()
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cell_along_minor_axis")
    def divideCellAlongMinorAxis(self, _cell):
        return self.divide_cell_along_minor_axis(cell=_cell)

    def divide_cell_along_minor_axis(self, cell):
        """
        Divides cell into two daughter cells along cleavage plane parallel to minor axis of the cell.
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a cell object that existed before division
        :param cell: {CompuCell.CellG} cell to divide
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisAlongMinorAxis(cell)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        # if self.mitosis_done:
        #     self.init_parent_and_child_cells()
        #     self.update_attributes()
        return mitosis_done


class MitosisSteppableClustersBase(SteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        SteppableBasePy.__init__(self, _simulator, _frequency)
        self.mitosisSteppable = CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parentCell = self.mitosisSteppable.parentCell
        self.childCell = self.mitosisSteppable.childCell
        self.mitosisDone = False

    def initParentAndChildCells(self):
        '''
        Initializes self.parentCell and self.childCell to point to respective cell objects after mitosis is completed succesfully
        '''
        self.parentCell = self.mitosisSteppable.parentCell
        self.childCell = self.mitosisSteppable.childCell

    def cloneClusterAttributes(self, sourceCellCluster, targetCellCluster, no_clone_key_dict_list=[]):
        for i in xrange(sourceCellCluster.size()):
            self.cloneAttributes(sourceCell=sourceCellCluster[i], targetCell=targetCellCluster[i],
                                 no_clone_key_dict_list=no_clone_key_dict_list)

    def cloneParentCluster2ChildCluster(self):
        # these calls seem to be necessary to ensure whatever is setin in mitosisSteppable (C++) is reflected in Python
        # self.parentCell=self.mitosisSteppable.parentCell
        # self.childCell=self.mitosisSteppable.childCell

        compartmentListParent = self.inventory.getClusterCells(self.parentCell.clusterId)
        compartmentListChild = self.inventory.getClusterCells(self.childCell.clusterId)

        self.cloneClusterAttributes(sourceCellCluster=compartmentListParent, targetCellCluster=compartmentListChild,
                                    no_clone_key_dict_list=[])

    def updateAttributes(self):
        parentCell = self.mitosisSteppable.parentCell
        childCell = self.mitosisSteppable.childCell

        compartmentListChild = self.inventory.getClusterCells(childCell.clusterId)
        compartmentListParent = self.inventory.getClusterCells(parentCell.clusterId)
        # compartments in the parent and child clusters arel listed in the same order so attribute changes require simple iteration through compartment list
        for i in xrange(compartmentListChild.size()):
            compartmentListChild[i].type = compartmentListParent[i].type

    def step(self, mcs):
        print
        "MITOSIS STEPPABLE Clusters BASE"

    def divideClusterRandomOrientation(self, _clusterId):
        self.mitosisDone = self.mitosisSteppable.doDirectionalMitosisRandomOrientationCompartments(_clusterId)
        if self.mitosisDone:
            self.initParentAndChildCells()
            self.updateAttributes()
        return self.mitosisDone

    def divideClusterOrientationVectorBased(self, _clusterId, _nx, _ny, _nz):
        self.mitosisDone = self.mitosisSteppable.doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId, _nx,
                                                                                                        _ny, _nz)
        if self.mitosisDone:
            self.initParentAndChildCells()
            self.updateAttributes()
        return self.mitosisDone

    def divideClusterAlongMajorAxis(self, _clusterId):
        # orientationVectors=self.mitosisSteppable.getOrientationVectorsMitosis(_cell)
        # print "orientationVectors.semiminorVec=",(orientationVectors.semiminorVec.fX,orientationVectors.semiminorVec.fY,orientationVectors.semiminorVec.fZ)
        self.mitosisDone = self.mitosisSteppable.doDirectionalMitosisAlongMajorAxisCompartments(_clusterId)
        if self.mitosisDone:
            self.initParentAndChildCells()
            self.updateAttributes()
        return self.mitosisDone

    def divideClusterAlongMinorAxis(self, _clusterId):
        self.mitosisDone = self.mitosisSteppable.doDirectionalMitosisAlongMinorAxisCompartments(_clusterId)
        if self.mitosisDone:
            self.initParentAndChildCells()
            self.updateAttributes()
        return self.mitosisDone
