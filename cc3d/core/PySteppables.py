import re
import itertools
from pathlib import Path
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
from typing import Union, Optional
from cc3d.cpp import CompuCell
from cc3d.core.SBMLSolverHelper import SBMLSolverHelper
from cc3d.core.MaBoSSCC3D import MaBoSSHelper
import types
import warnings
from deprecated import deprecated
from cc3d.core.SteeringParam import SteeringParam
from copy import deepcopy
from math import sqrt
from cc3d.core.numerics import *
from cc3d.core.Validation.sanity_checkers import validate_cc3d_entity_identifier


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
    def on_stop(self):
        """
        Called when simulation is stopped by user
        :return:
        """

    def cleanup(self):
        """

        :return:
        """


class FieldVisData:
    (CELL_LEVEL_SCALAR_FIELD, CELL_LEVEL_VECTOR_FIELD, HISTOGRAM) = list(range(0, 3))

    def __init__(self, field_name, field_type, attribute_name, function_obj=None):
        self.field = None
        self.field_name = field_name
        self.function_obj = function_obj
        self.attribute_name = attribute_name
        self.number_of_bins = 0
        self.cell_type_list = []
        if function_obj is None:
            self.function_obj = lambda x: x

        self.field_type = field_type


class PlotData:
    (HISTOGRAM) = list(range(0, 1))

    def __init__(self, plot_name, plot_type, attribute_name, function_obj=None):
        self.plot_name = plot_name
        self.plot_window = None
        self.function_obj = function_obj
        self.attribute_name = attribute_name
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.x_scale_type = 'linear'
        self.y_scale_type = 'linear'
        self.number_of_bins = 0
        self.color = 'green'
        self.cell_type_list = []

        if function_obj is None:
            self.function_obj = lambda x: x

        self.plot_type = plot_type


class CellTypeFetcher:
    def __init__(self, type_id_type_name_dict):
        # reversing dictionary from type_id_type_name_dict -> type_name_type_id_dict
        self.type_name_type_id_dict = {v: k for k, v in type_id_type_name_dict.items()}

    def __getattr__(self, item):

        try:
            return self.type_name_type_id_dict[item]
        except KeyError:
            raise KeyError(f'The requested cell type {item} does not exist')

    def get_data(self) -> dict:
        """
        Returns cell type data
        :return: dictionary mapping cell type name to cell type id
        """
        return self.type_name_type_id_dict


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


class SteppableBasePy(SteppablePy, SBMLSolverHelper, MaBoSSHelper):
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

        self.field = FieldFetcher()

        # plugin declarations - including legacy members
        self.neighbor_tracker_plugin = None
        self.neighborTrackerPlugin = None
        self.focal_point_plasticity_plugin = None
        self.focalPointPlasticityPlugin = None
        self.volume_tracker_plugin = None

        self._simulator = None

        # created and initialized in core_init method
        self.cell_type = None

        self.cell_field = None
        self.cellField = None
        self.cell_list = None
        self.cellList = None
        # cell_list_by_type is handled via a function call
        self.cellListByType = None
        self.cluster_list = None
        self.clusterList = None
        self.clusters = None

        self.mcs = -1
        # {plot_name:plotWindow  - pW object}
        self.plot_dict = {}
        self.shared_steppable_vars = {}

        # {field_name:FieldVisData } -  used to keep track of simple cell tracking visualizations
        self.tracking_field_vis_dict = {}

        # {field_name:PlotData } -  used to keep track of simple cell tracking plots
        self.tracking_plot_dict = {}

        self.plugin_init_dict = {
            "NeighborTracker": ['neighbor_tracker_plugin', 'neighborTrackerPlugin'],
            "FocalPointPlasticity": ['focal_point_plasticity_plugin', 'focalPointPlasticityPlugin'],
            "PixelTracker": ['pixel_tracker_plugin', 'pixelTrackerPlugin'],
            "BoundaryPixelTracker": ['boundary_pixel_tracker_plugin', 'boundaryPixelTrackerPlugin'],
            "BoundaryMonitor": ['boundary_monitor_plugin', 'boundaryMonitorPlugin'],
            "AdhesionFlex": ['adhesion_flex_plugin', 'adhesionFlexPlugin'],
            "PolarizationVector": ['polarization_vector_plugin', 'polarizationVectorPlugin'],
            "Polarization23": ['polarization_23_plugin', 'polarization23Plugin'],
            "CellOrientation": ['cell_orientation_plugin', 'cellOrientationPlugin'],
            "ContactOrientation": ['contact_orientation_plugin', 'contactOrientationPlugin'],
            "ContactLocalProduct": ['contact_local_product_plugin', 'contactLocalProductPlugin'],
            "ContactMultiCad": ['contact_multi_cad_plugin', 'contactMultiCadPlugin'],
            "LengthConstraint": ['length_constraint_plugin', 'lengthConstraintPlugin',
                                 'length_constraint_local_flex_plugin', 'lengthConstraintLocalFlexPlugin'],
            "ConnectivityGlobal": ['connectivity_global_plugin', 'connectivityGlobalPlugin'],
            "ConnectivityLocalFlex": ['connectivity_local_flex_plugin', 'connectivityLocalFlexPlugin'],
            "Chemotaxis": ['chemotaxis_plugin', 'chemotaxisPlugin'],
            "ClusterSurface": ['cluster_surface_plugin', 'clusterSurfacePlugin'],
            "ClusterSurfaceTracker": ['cluster_surface_tracker_plugin', 'clusterSurfaceTrackerPlugin'],
            "ElasticityTracker": ['elasticity_tracker_plugin', 'elasticityTrackerPlugin'],
            "PlasticityTracker": ['plasticity_tracker_plugin', 'plasticityTrackerPlugin'],
            "MomentOfInertia": ['moment_of_inertia_plugin', 'momentOfInertiaPlugin'],
            "OrientedGrowth": ['oriented_growth_plugin', 'orientedGrowthPlugin'],
            "Secretion": ["secretion_plugin", 'secretionPlugin']

        }

        # used by clone attributes functions
        self.clonable_attribute_names = ['lambdaVolume', 'targetVolume', 'targetSurface', 'lambdaSurface',
                                         'targetClusterSurface', 'lambdaClusterSurface', 'type', 'lambdaVecX',
                                         'lambdaVecY', 'lambdaVecZ', 'fluctAmpl']

    def merge_cells(self, source_cell, destination_cell):
        """
        Turns all voxels of source_cell into voxels of destination_cell
        :param source_cell: {CompuCell.CellG} cell to be "eaten"
        :param destination_cell: {CompuCell.CellG} cell "eating"
        :return:
        """

        source_vxs = self.get_cell_pixel_list(source_cell)
        if source_vxs is None:
            raise Exception("Couldn't fetch voxels of source_cell, did you load PixelTracker plugin?")

        for pixel_tracker_data in source_vxs:
            x, y, z = pixel_tracker_data.pixel.x, pixel_tracker_data.pixel.y, pixel_tracker_data.pixel.z
            self.cell_field[x, y, z] = destination_cell

    def cell_list_by_type(self, *args):
        """
        Returns a CellListByType object that represents list of ells with a given type
        :param args: list of cell types
        :return:
        """
        list_by_type_obj = CellListByType(self.inventory, *args)
        return list_by_type_obj

    def parameter_scan_main_output_folder(self) -> Optional[Path]:
        """
        Returns parameter scan main output folder. For example if instance of parameter scan is
        being written to
        C:/Users/m/CC3DWorkspace/CellSortingParameterScanWorkshop2020_output/scan_iteration_3/
        CellSortingParameterScanWorkshop2020
        it will return :
        C:/Users/m/CC3DWorkspace/CellSortingParameterScanWorkshop2020_output
        :return:
        """
        if self.output_dir is not None:
            param_scan_main_output_dir = Path(*Path(self.output_dir).parts[:-2])
            return param_scan_main_output_dir

    def open_file_in_parameter_scan_main_output_folder(self, file_name: str, mode: str = 'w') -> tuple:
        """
        opens file for writing in parameter scan main output folder. for example
        if instance of parameter scan is
        being written to
        C:/Users/m/CC3DWorkspace/CellSortingParameterScanWorkshop2020_output/scan_iteration_3/
        CellSortingParameterScanWorkshop2020
        it will open file - file_name in
        C:/Users/m/CC3DWorkspace/CellSortingParameterScanWorkshop2020_output
        :param file_name:
        :param mode:
        :return: tuple of (file_obj, full filepath)
        """
        parameter_scan_main_output_folder = self.parameter_scan_main_output_folder()
        if self.parameter_scan_main_output_folder() is not None:
            output_path = parameter_scan_main_output_folder.joinpath(file_name)

            return self.open_file(abs_path=output_path, mode=mode)

        return None, None

    @property
    def param_scan_iteration(self):
        return CompuCellSetup.persistent_globals.parameter_scan_iteration

    def open_file(self, abs_path: Union[Path, str], mode='w') -> tuple:
        """
        Opens file
        :param abs_path:
        :param mode:
        :return: tuple of (file_obj, full filepath)
        """
        output_path = Path(abs_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_handle = open(output_path, mode)
        except IOError:
            print(f"Could not open file: {abs_path} for writing")
            return None, None
        return file_handle, output_path

    def open_file_in_simulation_output_folder(self, file_name: str, mode: str = 'w') -> tuple:
        """
        attempts to open file in the simulation output folder

        :param file_name:
        :param mode:
        :return: tuple of (file_obj, full filepath)
        """
        if self.output_dir is not None:
            output_path = Path(self.output_dir).joinpath(file_name)
            return self.open_file(abs_path=output_path, mode=mode)

    @staticmethod
    def request_screenshot(mcs: int, screenshot_label: str) -> None:
        """
        Requests on-demand screenshot
        :param mcs:
        :param screenshot_label:
        :return:
        """
        pg = CompuCellSetup.persistent_globals
        screenshot_manager = pg.screenshot_manager
        screenshot_manager.add_ad_hoc_screenshot(mcs=mcs, screenshot_label=screenshot_label)

    def core_init(self, reinitialize_cell_types=True):
        """
        Performs actual initialization of members that point to C++ objects. This is function is called AFTER
        C++ objects are created and initialized. Therefore we cannot put those initialization calls in the steppable
        constructor because during Steppable construction C++ objects may not exist. This behavior is different
        thatn in previous versions of CC3D

        :param reinitialize_cell_types: {bool} indicates if types should be reinitialized or not . Sometimes core init
        might get called multiple times (e.g. during resize lattice event) and you donot want to reinitialize cell types

        :return:
        """

        self.cell_field = self.potts.getCellFieldG()
        self.cellField = self.cell_field
        self.cell_list = CellList(self.inventory)
        self.cellList = self.cell_list
        self.cellListByType = self.cell_list_by_type
        self.cluster_list = ClusterList(self.inventory)
        self.clusterList = self.cluster_list
        self.clusters = Clusters(self.inventory)

        persistent_globals = CompuCellSetup.persistent_globals
        persistent_globals.attach_dictionary_to_cells()

        type_id_type_name_dict = extract_type_names_and_ids()

        self.cell_type = CellTypeFetcher(type_id_type_name_dict=type_id_type_name_dict)

        if reinitialize_cell_types:
            for type_id, type_name in type_id_type_name_dict.items():
                self.typename_to_attribute(cell_type_name=type_name, type_id=type_id)
                # setattr(self, type_name.upper(), type_id)

        self.fetch_loaded_plugins()
        self.shared_steppable_vars = persistent_globals.shared_steppable_vars

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

    def track_cell_level_scalar_attribute(self, field_name: str, attribute_name: str, function_obj: object = None,
                                          cell_type_list: Union[list, None] = None):
        """
        Adds custom field that visualizes cell scalar attribute
        :param field_name:
        :param attribute_name:
        :param function_obj: function object that takes scalar and returns a scalar
        :param cell_type_list: list of cell types that should be tracked
        :return:
        """
        if cell_type_list is None:
            cell_type_list = []

        field_vis_data = FieldVisData(field_name=field_name,
                                      field_type=FieldVisData.CELL_LEVEL_SCALAR_FIELD,
                                      attribute_name=attribute_name,
                                      function_obj=function_obj)
        field_vis_data.cell_type_list = cell_type_list

        self.tracking_field_vis_dict[field_name] = field_vis_data

    def track_cell_level_vector_attribute(self, field_name: str, attribute_name: str, function_obj: object = None,
                                          cell_type_list: Union[list, None] = None):
        """

        Adds custom field that visualizes cell vector attribute
        :param field_name:
        :param attribute_name:
        :param function_obj: function object that takes vector and returns a vector
        :param cell_type_list: list of cell types that should be tracked
        :return:
        """

        if cell_type_list is None:
            cell_type_list = []

        field_vis_data = FieldVisData(field_name=field_name,
                                      field_type=FieldVisData.CELL_LEVEL_VECTOR_FIELD,
                                      attribute_name=attribute_name,
                                      function_obj=function_obj)
        field_vis_data.cell_type_list = cell_type_list

        self.tracking_field_vis_dict[field_name] = field_vis_data

    def histogram_scalar_attribute(self, histogram_name: str, attribute_name: str, number_of_bins: int,
                                   function: Union[object, None] = None,
                                   cell_type_list: Union[list, None] = None, x_axis_title: str = '',
                                   y_axis_title: str = '', color: str = 'green',
                                   x_scale_type: str = 'linear', y_scale_type: str = 'linear'):

        """
        Adds histogram that displays distribution of selected cell attribute
        :param histogram_name:
        :param attribute_name:
        :param number_of_bins:
        :param function:
        :param cell_type_list:
        :param x_axis_title:
        :param y_axis_title:
        :param color:
        :param x_scale_type:
        :param y_scale_type:
        :return:
        """

        if cell_type_list is None:
            cell_type_list = []

        tpd = PlotData(plot_name=histogram_name, plot_type=PlotData.HISTOGRAM, attribute_name=attribute_name,
                       function_obj=function)
        tpd.number_of_bins = number_of_bins

        tpd.x_scale_type = x_scale_type
        tpd.y_scale_type = y_scale_type

        tpd.x_axis_title = x_axis_title
        tpd.y_axis_title = y_axis_title
        if x_axis_title == '':
            tpd.x_axis_title = histogram_name
        if y_axis_title == '':
            tpd.y_axis_title = 'Value'

        tpd.color = color

        tpd.cell_type_list = cell_type_list

        self.tracking_plot_dict[histogram_name] = tpd

    def fetch_attribute(self, cell: object, attrib_name: str):
        """
        Fetches element of a dictionary attached to a cell or an attribute of cell (e.g.
        volume or lambdaSurface)
        :param cell:
        :param attrib_name:
        :return:
        """
        try:
            return cell.dict[attrib_name]
        except KeyError:
            try:
                return getattr(cell, attrib_name)
            except AttributeError:
                raise KeyError('Could not locate attribute: ' + attrib_name + ' in a cell object')

    def update_tracking_plots(self):

        for plot_name, tracking_plot_data in self.tracking_plot_dict.items():
            if tracking_plot_data.plot_type == PlotData.HISTOGRAM:
                self.update_tracking_histogram(tracking_plot_data)

    def update_tracking_histogram(self, tracking_plot_data):

        tpd = tracking_plot_data

        val_list = []
        plot_window = tpd.plot_window

        if not tpd.cell_type_list == []:
            selective_cell_list = self.cellList
        else:
            # unpacking list to positional arguments - using * operator
            selective_cell_list = self.cellListByType(*tpd.cell_type_list)
        try:
            for cell in selective_cell_list:

                try:
                    attrib = self.fetch_attribute(cell, tpd.attribute_name)
                except KeyError:
                    continue
                val_list.append(tpd.function_obj(attrib))

        except:
            raise RuntimeError('Automatic Attribute Tracking :wrong type of cell attribute, '
                               'missing attribute or wrong tracking function is used by track_cell_level functions')

        if len(val_list):
            plot_window.add_histogram(plot_name=tpd.plot_name, value_array=val_list, number_of_bins=tpd.number_of_bins)

    def update_tracking_fields(self):
        """
        Method that updates tracking fields that were initialized using track_cell_level... function
        :return: None
        """
        # tracking visualization part
        for field_name, field_vis_data in self.tracking_field_vis_dict.items():
            field_vis_data.field.clear()

            if field_vis_data.cell_type_list == []:
                selective_cell_list = self.cellList
            else:
                # unpacking lsit to positional arguments - using * operator
                selective_cell_list = self.cellListByType(*field_vis_data.cell_type_list)

            try:
                for cell in selective_cell_list:
                    try:
                        # attrib = cell.dict[field_vis_data.attribute_name]
                        attrib = self.fetch_attribute(cell, field_vis_data.attribute_name)

                    except KeyError:
                        continue
                    field_vis_data.field[cell] = field_vis_data.function_obj(attrib)
            except:
                raise RuntimeError(
                    'Automatic Attribute Tracking :'
                    'wrong type of cell attribute, '
                    'Missing attribute or wrong tracking function is used by track_cell_level functions')

    def initialize_automatic_tasks(self):
        self.initialize_tracking_fields()
        self.initialize_tracking_plots()

    def initialize_tracking_plots(self):

        for plot_name, tracking_plot_data in self.tracking_plot_dict.items():
            tpd = tracking_plot_data

            plot_win = self.add_new_plot_window(title='Histogram of Cell Volumes',
                                                x_axis_title='Number of Cells',
                                                y_axis_title='Volume Size in Pixels',
                                                x_scale_type=tpd.x_scale_type,
                                                y_scale_type=tpd.y_scale_type)

            plot_win.add_histogram_plot(plot_name=tpd.plot_name, color=tpd.color)
            tpd.plot_window = plot_win

    def initialize_tracking_fields(self):
        """

        :return:
        """
        for field_name, field_vis_data in self.tracking_field_vis_dict.items():
            # (CELL_LEVEL_SCALAR_FIELD, CELL_LEVEL_VECTOR_FIELD, HISTOGRAM)

            if field_vis_data.field_type == field_vis_data.CELL_LEVEL_SCALAR_FIELD:
                self.create_scalar_field_cell_level_py(field_name)
                field_vis_data.field = getattr(self.field, field_name)

            elif field_vis_data.field_type == field_vis_data.CELL_LEVEL_VECTOR_FIELD:
                self.create_vector_field_cell_level_py(field_name)
                field_vis_data.field = getattr(self.field, field_name)

    def perform_automatic_tasks(self):
        """
        Performs automatic tasks that normally woudl need to be called explicitely in tyhe steppale code
        updating plots at the end of steppable is one such task
        :return:
        """
        self.update_tracking_fields()
        self.update_tracking_plots()
        self.update_all_plots_windows()

    def update_all_plots_windows(self):
        """
        Updates all plots
        :return:
        """
        # tracking visualization part

        for plot_window_name, plot_window in self.plot_dict.items():
            plot_window.show_all_plots()
            # plot_window.showAllHistPlots()
            # plot_window.showAllBarCurvePlots()

    @staticmethod
    def set_output_dir(output_dir: str, abs_path: bool = False) -> None:
        """
        Sets output directory to output_dir. If  abs_path is False
        then the directory path will be w.r.t to workspace directory
        Otherwise it is expected that user provides absolute output path
        :param output_dir: directory name - relative (w.r.t to workspace dir) or absolute
        :param abs_path:  flag specifying if user provided absolute or relative path
        :return:
        """
        CompuCellSetup.set_output_dir(output_dir=output_dir, abs_path=abs_path)

    @property
    def output_dir(self):
        return CompuCellSetup.persistent_globals.output_directory

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

    @deprecated(version='4.0.0', reason="You should use : get_cluster_cells")
    def getClusterCells(self, _clusterId):
        return self.get_cluster_cells(cluster_id=_clusterId)

    def get_cluster_cells(self, cluster_id):
        """
        returns a container with cell objects that are members of compartmentalized cell (a cluster)
        :param cluster_id: {long int}
        :return:
        """
        return ClusterCellList(self.inventory.getClusterInventory().getClusterCells(cluster_id))

    def get_box_coordinates(self):
        return self.potts.getMinCoordinates(), self.potts.getMaxCoordinates()

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
        # NOTE: resetting of the dirty flag for the steering
        # panel model is done in the SteppableRegistry's "step" function

        if self.steering_param_dirty():
            self.process_steering_panel_data()

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
                try:
                    return CompuCellSetup.persistent_globals.steering_param_dict[name].dirty_flag
                except KeyError:
                    raise RuntimeError('Could not find steering_parameter named {}'.format(name))
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
                try:
                    pg.steering_param_dict[name].dirty_flag = flag
                except KeyError:
                    raise RuntimeError('Could not find steering_parameter named {}'.format(name))
            else:
                for p_name, steering_param in CompuCellSetup.persistent_globals.steering_param_dict.items():
                    steering_param.dirty_flag = flag

    def typename_to_attribute(self, cell_type_name: str, type_id: int) -> None:
        """
        sets steppable attribute based on type name
        Performs basic sanity checks
        :param cell_type_name:{str}
        :param type_id:{str}
        :return:
        """
        validate_cc3d_entity_identifier(cell_type_name, entity_type_label='cell type')
        cell_type_name_attr_list = [cell_type_name.upper(), f't_{cell_type_name}']

        for cell_type_name_attr in cell_type_name_attr_list:
            try:
                getattr(self, cell_type_name_attr)
                attribute_already_exists = True
            except AttributeError:
                attribute_already_exists = False

            if attribute_already_exists:
                raise AttributeError(
                    f'Could not convert cell type {cell_type_name} to steppable attribute. '
                    f'Attribute {cell_type_name_attr} already exists . Please change your cell type name')

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

        plot_win = CompuCellSetup.simulation_player_utils.add_new_plot_window(title, x_axis_title, y_axis_title,
                                                                              x_scale_type,
                                                                              y_scale_type, grid,
                                                                              config_options=config_options)
        self.plot_dict[title] = plot_win

        return plot_win

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

    def fetch_cell_by_id(self, cell_id: int) -> Union[None, CompuCell.CellG]:
        """
        Fetches cell by id. If cell does not exist it returns None
        :param cell_id: cell id
        :return: successfully fetched cell id or None
        """
        return self.inventory.attemptFetchingCellById(cell_id)

    @staticmethod
    def get_type_name_by_cell(_cell):
        if _cell is None:
            _type_id = 0
        else:
            _type_id = _cell.type

        type_id_type_name_dict = CompuCellSetup.simulation_utils.extract_type_names_and_ids()
        assert type_id_type_name_dict, "CellType plugin not found!"
        return type_id_type_name_dict[_type_id]

    @deprecated(version='4.0.0', reason="You should use : get_focal_point_plasticity_data_list")
    def getFocalPointPlasticityDataList(self, _cell):
        return self.get_focal_point_plasticity_data_list(cell=_cell)

    def get_focal_point_plasticity_data_list(self, cell):
        if self.focal_point_plasticity_plugin:
            return FocalPointPlasticityDataList(self.focal_point_plasticity_plugin, cell)

        return None

    @deprecated(version='4.0.0', reason="You should use : get_internal_focal_point_plasticity_data_list")
    def getInternalFocalPointPlasticityDataList(self, _cell):
        return self.get_internal_focal_point_plasticity_data_list(cell=_cell)

    def get_internal_focal_point_plasticity_data_list(self, cell):
        if self.focal_point_plasticity_plugin:
            return InternalFocalPointPlasticityDataList(self.focal_point_plasticity_plugin, cell)

        return None

    @deprecated(version='4.0.0', reason="You should use : get_anchor_focal_point_plasticity_data_list")
    def getAnchorFocalPointPlasticityDataList(self, _cell):
        return self.get_anchor_focal_point_plasticity_data_list(cell=_cell)

    def get_anchor_focal_point_plasticity_data_list(self, cell):
        if self.focal_point_plasticity_plugin:
            return AnchorFocalPointPlasticityDataList(self.focal_point_plasticity_plugin, cell)

        return None

    def get_focal_point_plasticity_neighbor_list(self, cell) -> []:
        """
        Return list of all cell objects linked to a cell
        :param cell: cell object for which to fetch list of linked cells
        :return: list of linked cells
        """
        if self.focal_point_plasticity_plugin is None:
            return []
        else:
            return [c for c in self.get_fpp_linked_cells(cell)]

    def get_focal_point_plasticity_internal_neighbor_list(self, cell) -> []:
        """
        Return list of all cell objects linked to a cell
        :param cell: cell object for which to fetch list of linked cells
        :return: list of linked cells
        """
        if self.focal_point_plasticity_plugin is None:
            return []
        else:
            return [c for c in self.get_fpp_internal_linked_cells(cell)]

    def get_focal_point_plasticity_num_neighbors(self, cell) -> int:
        """
        Returns number of all cell objects linked to a cell
        :param cell: cell object for which to count linked cells
        :return: number of linked cells
        """
        return len(self.get_fpp_linked_cells(cell))

    def get_focal_point_plasticity_num_internal_neighbors(self, cell) -> int:
        """
        Returns number of all cell objects internally linked to a cell
        :param cell: cell object for which to count internally linked cells
        :return: number of internally linked cells
        """
        return len(self.get_fpp_internal_linked_cells(cell))

    def get_focal_point_plasticity_is_linked(self, cell1, cell2) -> bool:
        """
        Returns if two cells are linked
        :param cell1: first cell object
        :param cell2: second cell object
        :return: True if cells are linked
        """
        if self.focal_point_plasticity_plugin is None:
            return False
        else:
            return self.get_fpp_link_by_cells(cell1, cell2) is not None

    def get_focal_point_plasticity_is_internally_linked(self, cell1, cell2) -> bool:
        """
        Returns if two cells are internally linked
        :param cell1: first cell object
        :param cell2: second cell object
        :return: True if cells are internally linked
        """
        if self.focal_point_plasticity_plugin is None:
            return False
        else:
            return self.get_fpp_internal_link_by_cells(cell1, cell2) is not None

    def get_focal_point_plasticity_initiator(self, cell1, cell2):
        """
        Returns which cell initiated a link; returns None if cells are not linked
        :param cell1: first cell object in link
        :param cell2: second cell object in link
        :return: cell that initiated the link, or None if cells are not linked
        """
        if not self.get_focal_point_plasticity_is_linked(cell1=cell1, cell2=cell2):
            return None
        else:
            link = self.get_fpp_link_by_cells(cell1, cell2)
            if link is None:
                return None
            return link.getObj0()

    def get_focal_point_plasticity_internal_initiator(self, cell1, cell2):
        """
        Returns which cell initiated an internal link; returns None if cells are not linked
        :param cell1: first cell object in internal link
        :param cell2: second cell object in internal link
        :return: cell that initiated the internal link, or None if cells are not linked
        """
        if not self.get_focal_point_plasticity_is_linked(cell1=cell1, cell2=cell2):
            return None
        else:
            link = self.get_fpp_internal_link_by_cells(cell1, cell2)
            if link is None:
                return None
            return link.getObj0()

    def set_focal_point_plasticity_parameters(self, cell, n_cell=None, lambda_distance: float = None,
                                              target_distance: float = None, max_distance: float = None) -> None:
        """
        Sets focal point plasticity parameters for a cell; unspecified parameters are unchanged
        :param cell: cell object for which to modify parameters
        :param n_cell: linked cell object describing link
        sets parameters for all links attached to cell if n_cell is None
        :param lambda_distance: Lagrange multiplier for link(s)
        :param target_distance: target distance of link(s)
        :param max_distance: maximum distance of link(s)
        :return: None
        """
        if self.focal_point_plasticity_plugin is None:
            return

        if n_cell is None:
            linked_list = self.get_fpp_linked_cells(cell)
        else:
            linked_list = [self.get_fpp_link_by_cells(cell, n_cell)]
        for link in linked_list:
            if lambda_distance is not None:
                link.setLambdaDistance(lambda_distance)
            if target_distance is not None:
                link.setTargetDistance(target_distance)
            if max_distance is not None:
                link.setMaxDistance(max_distance)

    @property
    def fpp_link_inventory(self):
        assert self.focal_point_plasticity_plugin is not None, 'Load focal point plasticity plugin'
        return self.focal_point_plasticity_plugin.getLinkInventory()

    @property
    def fpp_internal_link_inventory(self):
        assert self.focal_point_plasticity_plugin is not None, 'Load focal point plasticity plugin'
        return self.focal_point_plasticity_plugin.getInternalLinkInventory()

    @property
    def fpp_anchor_inventory(self):
        assert self.focal_point_plasticity_plugin is not None, 'Load focal point plasticity plugin'
        return self.focal_point_plasticity_plugin.getAnchorInventory()

    def get_focal_point_plasticity_link_list(self):
        """
        Returns list of all links
        :return: {list} list of all links
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        return FocalPointPlasticityLinkList(self.focal_point_plasticity_plugin)

    def get_focal_point_plasticity_internal_link_list(self):
        """
        Returns list of all internal links
        :return: {list} list of all internal links
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        return FocalPointPlasticityInternalLinkList(self.focal_point_plasticity_plugin)

    def get_focal_point_plasticity_anchor_list(self):
        """
        Returns list of all anchors
        :return: {list} list of all anchors
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        return FocalPointPlasticityAnchorList(self.focal_point_plasticity_plugin)

    def new_fpp_link(self, initiator: CompuCell.CellG, initiated: CompuCell.CellG,
                     lambda_distance: float, target_distance: float = 0.0, max_distance: float = 0.0):
        """
        Create a focal point plasticity link
        :param initiator: {CompuCell.CellG} cell initiating the link
        :param initiated: {CompuCell.CellG} second cell of the link
        :param lambda_distance: {float} link lambda coefficient
        :param target_distance: {float} target link distance
        :param max_distance: {float} maximum link distance
        :return: {CompuCell.FocalPointPlasticityLink} created link
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        self.focal_point_plasticity_plugin.createFocalPointPlasticityLink(
            initiator, initiated, lambda_distance, target_distance, max_distance)
        return self.fpp_link_inventory.getLinkByCells(initiator, initiated)

    def new_fpp_internal_link(self, initiator: CompuCell.CellG, initiated: CompuCell.CellG,
                              lambda_distance: float, target_distance: float = 0.0, max_distance: float = 0.0):
        """
        Create a focal point plasticity internal link
        :param initiator: {CompuCell.CellG} cell initiating the link
        :param initiated: {CompuCell.CellG} second cell of the link
        :param lambda_distance: {float} link lambda coefficient
        :param target_distance: {float} target link distance
        :param max_distance: {float} maximum link distance
        :return: {CompuCell.FocalPointPlasticityInternalLink} created link
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        self.focal_point_plasticity_plugin.createInternalFocalPointPlasticityLink(
            initiator, initiated, lambda_distance, target_distance, max_distance)
        return self.fpp_internal_link_inventory.getLinkByCells(initiator, initiated)

    def new_fpp_anchor(self, cell: CompuCell.CellG,
                       lambda_distance: float, target_distance: float = 0.0, max_distance: float = 0.0,
                       x: float = 0.0, y: float = 0.0, z: float = 0.0, pt: CompuCell.Point3D = None):
        """
        Create a focal point plasticity internal link
        :param cell: {CompuCell.CellG} cell
        :param lambda_distance: {float} link lambda coefficient
        :param target_distance: {float} target link distance
        :param max_distance: {float} maximum link distance
        :param x: {float} x-coordinate of anchor
        :param y: {float} y-coordinate of anchor
        :param z: {float} z-coordinate of anchor
        :param pt: {CompuCell3D.Point3D} anchor point
        :return: {CompuCell.FocalPointPlasticityAnchor} created link
        """
        if self.focal_point_plasticity_plugin is None:
            return None
        if pt is not None:
            x, y, z = pt.x, pt.y, pt.z
        anchor_id = int(self.focal_point_plasticity_plugin.createAnchor(
            cell, lambda_distance, target_distance, max_distance, x, y, z))
        return self.fpp_anchor_inventory.getAnchor(cell, anchor_id)

    def delete_fpp_link(self, _link):
        """
        Deletes a focal point plasticity link, internal link or anhcor
        :param _link: link, internal link or anchor
        :return: None
        """
        assert self.focal_point_plasticity_plugin is not None, 'Load focal point plasticity plugin'
        if isinstance(_link, CompuCell.FocalPointPlasticityLink):
            self.focal_point_plasticity_plugin.deleteFocalPointPlasticityLink(_link.getObj0(), _link.getObj1())
        elif isinstance(_link, CompuCell.FocalPointPlasticityInternalLink):
            self.focal_point_plasticity_plugin.deleteInternalFocalPointPlasticityLink(_link.getObj0(), _link.getObj1())
        elif isinstance(_link, CompuCell.FocalPointPlasticityAnchor):
            self.focal_point_plasticity_plugin.deleteAnchor(_link.getObj0(), _link.getAnchorId())

    def remove_all_cell_fpp_links(self, _cell: CompuCell.CellG,
                                  links: bool = False, internal_links: bool = False, anchors: bool = False):
        """
        Removes all links associated with a cell; if no optional arguments are specified, then all links are removed
        :param _cell: {CompuCell.CellG} cell
        :param links: {bool} option for fpp links
        :param internal_links: {bool} option for fpp internal links
        :param anchors: {bool} option for fpp anchors
        :return: None
        """
        assert self.focal_point_plasticity_plugin is not None, 'Load focal point plasticity plugin'
        if not any([links, internal_links, anchors]):
            links, internal_links, anchors = True, True, True
        if links:
            self.fpp_link_inventory.removeCellLinks(_cell)
        if internal_links:
            self.fpp_internal_link_inventory.removeCellLinks(_cell)
        if anchors:
            self.fpp_anchor_inventory.removeCellLinks(_cell)

    def get_number_of_fpp_links(self) -> int:
        """
        Returns number of links
        :return: {int} number of links
        """
        return int(self.fpp_link_inventory.getLinkInventorySize())

    def get_number_of_fpp_internal_links(self) -> int:
        """
        Returns number of internal links
        :return: {int} number of internal links
        """
        return int(self.fpp_internal_link_inventory.getLinkInventorySize())

    def get_number_of_fpp_anchors(self) -> int:
        """
        Returns number of anchors
        :return: {int} number of anchors
        """
        return int(self.fpp_anchor_inventory.getLinkInventorySize())

    def get_fpp_link_by_cells(self, cell1: CompuCell.CellG, cell2: CompuCell.CellG) -> CompuCell.FocalPointPlasticityLink:
        """
        Returns link associated with two cells
        :param cell1: first cell
        :param cell2: second cell
        :return: {CompuCell.FocalPointPlasticityLink} link
        """
        return self.fpp_link_inventory.getLinkByCells(cell1, cell2)

    def get_fpp_internal_link_by_cells(self, cell1: CompuCell.CellG, cell2: CompuCell.CellG) -> CompuCell.FocalPointPlasticityInternalLink:
        """
        Returns internal link associated with two cells
        :param cell1: first cell
        :param cell2: second cell
        :return: {CompuCell.FocalPointPlasticityInternalLink} internal link
        """
        return self.fpp_internal_link_inventory.getLinkByCells(cell1, cell2)

    def get_fpp_anchor_by_cell_and_id(self, cell: CompuCell.CellG, anchor_id: int) -> CompuCell.FocalPointPlasticityAnchor:
        """
        Returns internal link associated with two cells
        :param cell: cell
        :param anchor_id: anchor id
        :return: {CompuCell.FocalPointPlasticityAnchor} anchor
        """
        return self.fpp_anchor_inventory.getAnchor(cell, anchor_id)

    def get_fpp_links_by_cell(self, _cell: CompuCell.CellG) -> CompuCell.FPPLinkList:
        """
        Get list of links by cell
        :param _cell: cell
        :return: links
        """
        return self.fpp_link_inventory.getCellLinkList(_cell)

    def get_fpp_internal_links_by_cell(self, _cell: CompuCell.CellG) -> CompuCell.FPPInternalLinkList:
        """
        Get list of internal links by cell
        :param _cell: cell
        :return: internal links
        """
        return self.fpp_internal_link_inventory.getCellLinkList(_cell)

    def get_fpp_anchors_by_cell(self, _cell: CompuCell.CellG) -> CompuCell.FPPAnchorList:
        """
        Get list of anchors by cell
        :param _cell: cell
        :return: anchors
        """
        return self.fpp_anchor_inventory.getCellLinkList(_cell)

    def get_fpp_linked_cells(self, _cell: CompuCell.CellG) -> CompuCell.mvectorCellGPtr:
        """
        Get list of cells linked to a cell
        :param _cell: cell
        :return: list of linked cells
        """
        return self.fpp_link_inventory.getLinkedCells(_cell)

    def get_fpp_internal_linked_cells(self, _cell: CompuCell.CellG) -> CompuCell.mvectorCellGPtr:
        """
        Get list of cells internally linked to a cell
        :param _cell: cell
        :return: list of linked cells
        """
        return self.fpp_internal_link_inventory.getLinkedCells(_cell)

    def get_number_of_fpp_junctions_by_type(self, _cell: CompuCell.CellG, _type: int) -> int:
        """
        Get number of link junctions by type for a cell
        :param _cell: cell
        :param _type: type id
        :return: {int} number of junctions
        """
        return int(self.fpp_link_inventory.getNumberOfJunctionsByType(_cell, _type))

    def get_number_of_fpp_internal_junctions_by_type(self, _cell: CompuCell.CellG, _type: int) -> int:
        """
        Get number of internal link junctions by type for a cell
        :param _cell: cell
        :return: {int} number of internal junctions
        """
        return int(self.fpp_internal_link_inventory.getNumberOfJunctionsByType(_cell, _type))

    def get_energy_calculations(self):
        return EnergyDataList(self.potts)

    @deprecated(version='4.0.0', reason="You should use : get_elasticity_data_list")
    def getElasticityDataList(self, _cell):
        return self.get_elasticity_data_list(cell=_cell)

    def get_elasticity_data_list(self, cell):
        if self.elasticity_tracker_plugin:
            return ElasticityDataList(self.elasticity_tracker_plugin, cell)

    @deprecated(version='4.0.0', reason="You should use : get_plasticity_data_list")
    def getPlasticityDataList(self, _cell):
        return self.get_plasticity_data_list(cell=_cell)

    def get_plasticity_data_list(self, cell):
        if self.plasticity_tracker_plugin:
            return PlasticityDataList(self.plasticity_tracker_plugin, cell)

    @deprecated(version='4.0.0', reason="You should use : build_wall")
    def buildWall(self, type):
        return self.build_wall(cell_type=type)

    def build_wall(self, cell_type):
        """
        builds a 1px wide wall around the lattice
        :param cell_type: type pf cells that will make up the wall. This should be a frozen cell type
        :return:
        """
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
                self.cell_field[self.dim.x - 1:self.dim.x, 0:self.dim.y, 0] = cell

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
        """
        removes wall around the lattice
        :return:
        """
        # build wall of Medium
        self.build_wall(0)

    @deprecated(version='4.0.0', reason="You should use : change_number_of_work_nodes")
    def changeNumberOfWorkNodes(self, _numberOfWorkNodes):
        return self.change_number_of_work_nodes(number_of_work_nodes=_numberOfWorkNodes)

    def change_number_of_work_nodes(self, number_of_work_nodes: int):
        """
        changes number of CPU's while simulation runs
        :param number_of_work_nodes:
        :return:
        """

        number_of_work_nodes_ev = CompuCell.CC3DEventChangeNumberOfWorkNodes()
        number_of_work_nodes_ev.oldNumberOfNodes = 1
        number_of_work_nodes_ev.newNumberOfNodes = number_of_work_nodes
        self.simulator.postEvent(number_of_work_nodes_ev)

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

        # need to store simulator in a tmp variable because it is overwritten (set to None) in self.__init__
        simulator = self._simulator

        self.__init__(self.frequency)

        # restoring self._simulator
        self._simulator = simulator

        self.core_init(reinitialize_cell_types=False)

        # with new cell field and possibly other fields  we have to reinitialize steppables
        for steppable in CompuCellSetup.persistent_globals.steppable_registry.allSteppables():
            if steppable != self:
                steppable_simulator = None
                if hasattr(steppable, '_simulator'):
                    steppable_simulator = steppable._simulator

                steppable.__init__(steppable.frequency)

                if steppable_simulator is not None:
                    steppable._simulator = steppable_simulator
                    if hasattr(steppable, 'core_init'):
                        steppable.core_init(reinitialize_cell_types=False)

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
        Distance between two points. Respects boundary conditions
        :return: {float} invariant distance between two points
        """

        return self.vector_norm(self.invariant_distance_vector(p1, p2))

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector_integer")
    def invariantDistanceVectorInteger(self, _from, _to):
        return self.invariant_distance_vector_integer(p1=_from, p2=_to)

    def invariant_distance_vector_integer(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Point3D points. respects
        Boundary conditions
        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """
        boundary_strategy = CompuCell.BoundaryStrategy.getInstance()
        dist_vec = CompuCell.distanceVectorInvariant(p2, p1, self.dim, boundary_strategy)
        return np.array([float(dist_vec.x), float(dist_vec.y), float(dist_vec.z)])

    @deprecated(version='4.0.0', reason="You should use : unconditional_invariant_distance_vector_integer")
    def unconditionalInvariantDistanceVectorInteger(self, _from, _to):
        return self.unconditional_invariant_distance_vector_integer(p1=_from, p2=_to)

    def unconditional_invariant_distance_vector_integer(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Point3D points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place.
        The reason we call it unconditional is because invariant distance this function computes assumes we have
        periodic boundary conditions in place irrespective if this is true or not. For some applications this
        function may be inappropriate. It is appropriate if the two points we are computing distance between are
        relatively close

        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        dist_vec = CompuCell.unconditionalDistanceVectorInvariant(p2, p1, self.dim)
        return np.array([float(dist_vec.x), float(dist_vec.y), float(dist_vec.z)])

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector")
    def invariantDistanceVector(self, _from, _to):
        return self.invariant_distance_vector(p1=_from, p2=_to)

    def invariant_distance_vector(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Coordinates3D<double> points
        Respects boundary conditions
        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        boundary_strategy = CompuCell.BoundaryStrategy.getInstance()
        dist_vec = CompuCell.distanceVectorCoordinatesInvariant(p2, p1, self.dim, boundary_strategy)
        return np.array([dist_vec.x, dist_vec.y, dist_vec.z])

    @deprecated(version='4.0.0', reason="You should use : unconditional_invariant_distance_vector")
    def unconditionalInvariantDistanceVector(self, _from, _to):

        return self.unconditional_invariant_distance_vector(p1=_from, p2=_to)

    def unconditional_invariant_distance_vector(self, p1, p2):
        """
        This function will calculate distance vector with integer coordinates between two Coordinates3D<double> points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place
        :param p1: {list} position of first point
        :param p2: {list} position of second point
        :return: {ndarray} distance vector
        """

        dist_vec = CompuCell.unconditionalDistanceVectorCoordinatesInvariant(p2, p1, self.dim)
        return np.array([dist_vec.x, dist_vec.y, dist_vec.z])

    @deprecated(version='4.0.0', reason="You should use : vector_norm")
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
        It is the most straightforward way to compute distance
        :return: {ndarray} distance vector
        """
        return self.distance_vector([cell1.xCOM, cell1.yCOM, cell1.zCOM], [cell2.xCOM, cell2.yCOM, cell2.zCOM])

    @deprecated(version='4.0.0', reason="You should use : invariant_distance_vector_between_cells")
    def invariantDistanceVectorBetweenCells(self, _cell_from, _cell_to):
        return self.invariant_distance_vector_between_cells(cell1=_cell_from, cell2=_cell_to)

    def invariant_distance_vector_between_cells(self, cell1, cell2):
        """
        This function will calculate distance vector between  COM's of cells . Respects boundary conditions
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

    def cell_velocity(self, _cell: CompuCell.CellG) -> CompuCell.Coordinates3DDouble:
        """
        Get the velocity of a cell, in units lattice sites / step

        Note that this method is slightly slower than manually calculating velocity from cell attributes
        but is safe for periodic boundary conditions.

        To manually perform the same calculations, do something like the following,

        .. code-block:: python

            cell: CompuCell.CellG
            vx, vy, vz = cell.xCOM - cell.xCOMPrev, cell.yCOM - cell.yCOMPrev, cell.zCOM - cell.zCOMPrev
            cell_velocity = CompuCell.Coordinates3DDouble(vx, vy, vz)

        :param _cell: cell
        :type _cell: CompuCell.CellG
        :return: instantaneous velocity of the cell at its center of mass
        :rtype: CompuCell.Coordinates3DDouble
        """
        boundary_strategy = CompuCell.BoundaryStrategy.getInstance()
        return CompuCell.cellVelocity(_cell, self.dim, boundary_strategy)

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

    @deprecated(version='4.0.0', reason="You should use : get_cell_boundary_pixel_list")
    def getCellBoundaryPixelList(self, _cell, _neighborOrder=-1):
        return self.get_cell_boundary_pixel_list(cell=_cell, neighbor_order=_neighborOrder)

    def get_cell_boundary_pixel_list(self, cell, neighbor_order=-1):
        if self.boundaryPixelTrackerPlugin:
            return CellBoundaryPixelList(self.boundaryPixelTrackerPlugin, cell, neighbor_order)

        return None

    @deprecated(version='4.0.0', reason="You should use : move_cell")
    def moveCell(self, cell, shiftVector):
        return self.move_cell(cell=cell, shift_vector=shiftVector)

    def point3d_to_tuple(self, pt: CompuCell.Point3D) -> tuple:
        """
        Converts CompuCell.Point3D into tuple
        :param pt:
        :return:
        """
        return pt.x, pt.y, pt.z

    def move_cell(self, cell, shift_vector):
        """
        Moves cell by shift_vector
        :param cell: {CompuCell.CellG} cell
        :param shift_vector: {tuple,  list, array, or CompuCell.Point3D} 3-element list specifying shoft vector
        :return: None
        """

        if not cell:
            raise TypeError(f'Cannot move non existing cell. Expected cell to be CompuCell.CellG, got {type(cell)}')

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
        if pixel_list is None:
            raise AttributeError('Could not find PixelTracker Plugin')
        pt = CompuCell.Point3D()

        # we have to make two sets (for faster lookup) of pixels :
        # set used to hold pixels to delete
        pixels_to_delete_set = set()
        # set used to hold pixels to move
        pixels_to_move_set = set()
        for pixel_tracker_data in pixel_list:
            pt.x = pixel_tracker_data.pixel.x + shift_vec.x
            pt.y = pixel_tracker_data.pixel.y + shift_vec.y
            pt.z = pixel_tracker_data.pixel.z + shift_vec.z
            # here we are making a copy of the cell
            pixels_to_delete_set.add(self.point3d_to_tuple(pixel_tracker_data.pixel))

            if self.check_if_in_the_lattice(pt):
                pixels_to_move_set.add(self.point3d_to_tuple(pt))

        # Now we will move cell
        for pt_tuple in pixels_to_move_set:
            self.cell_field[pt_tuple] = cell

        # Now we will delete old pixels
        medium_cell = CompuCell.getMediumCell()
        for pixel_tuple in pixels_to_delete_set:
            # Safe deletion. Don't delete the old pixel
            # if it is part of the new pixel set
            if pixel_tuple not in pixels_to_move_set:
                self.cell_field[pixel_tuple] = medium_cell

    @deprecated(version='4.0.0', reason="You should use : check_if_in_the_lattice")
    def checkIfInTheLattice(self, _pt):
        return self.check_if_in_the_lattice(pt=_pt)

    def check_if_in_the_lattice(self, pt):
        if pt.x >= 0 and pt.x < self.dim.x and pt.y >= 0 and pt.y < self.dim.y and pt.z >= 0 and pt.z < self.dim.z:
            return True
        return False

    @deprecated(version='4.0.0', reason="You should use : get_copy_of_cell_pixels")
    def getCopyOfCellPixels(self, _cell, _format=CC3D_FORMAT):
        return self.get_copy_of_cell_pixels(cell=_cell, format=_format)

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

    @deprecated(version='4.0.0', reason="You should use : get_copy_of_cell_boundary_pixels")
    def getCopyOfCellBoundaryPixels(self, _cell, _format=CC3D_FORMAT):

        return self.get_copy_of_cell_boundary_pixels(cell=_cell, format=_format)

    def get_copy_of_cell_boundary_pixels(self, cell, format=CC3D_FORMAT):

        try:
            if format == SteppableBasePy.CC3D_FORMAT:
                return [CompuCell.Point3D(boundaryPixelTrackerData.pixel) for boundaryPixelTrackerData in
                        self.getCellBoundaryPixelList(cell)]
            else:
                return [(boundaryPixelTrackerData.pixel.x, boundaryPixelTrackerData.pixel.y,
                         boundaryPixelTrackerData.pixel.z) for boundaryPixelTrackerData in
                        self.getCellBoundaryPixelList(cell)]
        except:
            raise AttributeError('Could not find BoundaryPixelTracker Plugin')

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

    @deprecated(version='4.0.0', reason="You should use : clone_attributes")
    def cloneAttributes(self, sourceCell, targetCell, no_clone_key_dict_list=None):
        if no_clone_key_dict_list is None:
            no_clone_key_dict_list = []

        return self.clone_attributes(source_cell=sourceCell, target_cell=targetCell,
                                     no_clone_key_dict_list=no_clone_key_dict_list)

    def clone_attributes(self, source_cell, target_cell, no_clone_key_dict_list=None):
        """
        Copies attributes from source cell to target cell. Users can specify which attributes should not be clones
        using no_clone_key_dict_list
        :param source_cell: {CompuCell.CellG} source cell
        :param target_cell: {CompuCell.CellG} target cell
        :param no_clone_key_dict_list: {list} list of dictionaries of attributes that are not to be cloned
        :return:
        """
        if no_clone_key_dict_list is None:
            no_clone_key_dict_list = []

        # clone "C++" attributes
        for attrName in self.clonable_attribute_names:
            setattr(target_cell, attrName, getattr(source_cell, attrName))

        # clone dictionary
        for key, val in source_cell.dict.items():

            if key in no_clone_key_dict_list:
                continue
            elif key == '__sbml_fetcher':
                # we are skipping copying of SWIG-added attribute
                # SBMLFetcher - this is added by default during cell creation
                # co no need to copy
                continue
            elif key == 'SBMLSolver':
                self.copy_sbml_simulators(from_cell=source_cell, to_cell=target_cell)
            elif key == CompuCell.CellG.__maboss__:
                # skipping MaBoSS models; need a reliable copy constructor
                continue
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

    @deprecated(version='4.0.0', reason="You should use : reassign_cluster_id")
    def reassignClusterId(self, _cell, _clusterId):
        return self.reassign_cluster_id(cell=_cell, cluster_id=_clusterId)

    def reassign_cluster_id(self, cell, cluster_id):
        """
        reassigns cluster id of cells
        :param cell: {cell}
        :param cluster_id:{int} new cluster id
        :return:
        """
        old_cluster_id = cell.clusterId
        new_cluster_id = cluster_id
        self.inventory.reassignClusterId(cell, new_cluster_id)
        if self.clusterSurfaceTrackerPlugin:
            self.clusterSurfaceTrackerPlugin.updateClusterSurface(old_cluster_id)
            self.clusterSurfaceTrackerPlugin.updateClusterSurface(new_cluster_id)

    @deprecated(version='4.0.0', reason="You should use : get_field_secretor")
    def getFieldSecretor(self, _fieldName):
        return self.get_field_secretor(field_name=_fieldName)

    def get_field_secretor(self, field_name):

        if self.secretionPlugin:
            return self.secretionPlugin.getFieldSecretor(field_name)

        raise RuntimeError(
            "Please define Secretion Plugin in the XML before requesting secretor object from Python script."
            'Secreion Plugin can be defined by including <Plugin Name="Secretion"/> in the XML')

    @deprecated(version='4.0.0', reason="You should use : hex_2_cartesian")
    def hex2Cartesian(self, _in):
        return self.hex_2_cartesian(coords=_in)

    def hex_2_cartesian(self, coords):
        """
        this transformation takes coordinates of a point on a hex lattice and returns integer coordinates of cartesian
        pixel that is nearest given point on hex lattice
        It is the inverse transformation of the one coded in HexCoord in BoundaryStrategy.cpp (see Hex2Cartesian).

        Argument: _in is either a tuple or a list or array with 3 elements or Coordinates3D<double> object
        returns Point3D
        """
        bs = self.simulator.getBoundaryStrategy()
        return bs.Hex2Cartesian(coords)

    @deprecated(version='4.0.0', reason="You should use : hex_2_cartesian_python")
    def hex2CartesianPython(self, _in):
        return self.hex_2_cartesian_python(coords=_in)

    def hex_2_cartesian_python(self, coords):
        '''
        this transformation takes coordinates of a point on ahex lattice and returns integer coordinates of cartesian
        pixel that is nearest given point on hex lattice
        It is the inverse transformation of the one coded in HexCoord in BoundaryStrategy.cpp.
        NOTE: there is c+= implementation of this function which is much faster and
        Argument: _in is either a tuple or a list or array with 3 elements
        '''

        def ir(x):
            return int(round(x))

        z_segments = ir(coords[2] / (sqrt(6.0) / 3.0))

        if (z_segments % 3) == 1:
            y_segments = ir(coords[1] / (sqrt(3.0) / 2.0) - 2.0 / 6.0)

            if y_segments % 2:

                return CompuCell.Point3D(ir(coords[0] - 0.5), y_segments, z_segments)

            else:

                return CompuCell.Point3D(ir(coords[0]), y_segments, z_segments)

        elif (z_segments % 3) == 2:

            y_segments = ir(coords[1] / (sqrt(3.0) / 2.0) + 2.0 / 6.0)

            if y_segments % 2:
                return CompuCell.Point3D(ir(coords[0] - 0.5), y_segments, z_segments)
            else:
                return CompuCell.Point3D(ir(coords[0]), y_segments, z_segments)

        else:
            y_segments = ir(coords[1] / (sqrt(3.0) / 2.0))

            if y_segments % 2:
                return CompuCell.Point3D(ir(coords[0]), y_segments, z_segments)
            else:
                return CompuCell.Point3D(ir(coords[0] - 0.5), y_segments, z_segments)

    @deprecated(version='4.0.0', reason="You should use : cartesian_2_hex")
    def cartesian2Hex(self, _in):
        return self.cartesian_2_hex(coords=_in)

    def cartesian_2_hex(self, coords):
        """
        this transformation takes coordinates of a point on a cartesian lattice and returns hex coordinates
        It is coded as HexCoord fcn in BoundaryStrategy.cpp.
        NOTE: there is c++ implementation of this function which is much faster and
        Argument: _in is either a tuple or a list or array with 3 elements or Point3D object
        returns Coordinates3D<double>
        """
        bs = self.simulator.getBoundaryStrategy()
        return bs.HexCoord(coords)

    @deprecated(version='4.0.0', reason="You should use : point_3d_to_numpy")
    def point3DToNumpy(self, _pt):
        return self.point_3d_to_numpy(pt=_pt)

    def point_3d_to_numpy(self, pt):
        """
        This function converts CompuCell.Point3D into floating point numpy array(vector) of size 3
        """

        return np.array([float(pt.x), float(pt.y), float(pt.z)])

    @deprecated(version='4.0.0', reason="You should use : numpy_to_point_3d")
    def numpyToPoint3D(self, _array):
        return self.numpy_to_point_3d(array=_array)

    def numpy_to_point_3d(self, array):

        pt = CompuCell.Point3D()
        pt.x = array[0]
        pt.y = array[1]
        pt.z = array[2]
        return pt

    @deprecated(version='4.0.0', reason="You should use : are_cells_different")
    def areCellsDifferent(self, _cell1, _cell2):
        return self.are_cells_different(cell1=_cell1, cell2=_cell2)

    def are_cells_different(self, cell1, cell2):
        """
        Checks if two cells are different
        :param cell1: {CellG c++ obj}
        :param cell2: {CellG c++ obj}
        :return: {bool}
        """
        return CompuCell.areCellsDifferent(cell1, cell2)

    @deprecated(version='4.0.0', reason="You should use : set_max_mcs")
    def setMaxMCS(self, maxMCS):
        return self.set_max_mcs(max_mcs=maxMCS)

    def set_max_mcs(self, max_mcs):
        self.simulator.setNumSteps(max_mcs)


class MitosisSteppableBase(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.mitosisSteppable = None
        self.parent_cell = None
        self.child_cell = None
        # legacy API
        self.parentCell = None
        self.childCell = None

        self._parent_child_position_flag = None

    def core_init(self, reinitialize_cell_types=True):
        SteppableBasePy.core_init(self, reinitialize_cell_types=reinitialize_cell_types)
        self.mitosisSteppable = CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parent_cell = self.mitosisSteppable.parentCell
        self.child_cell = self.mitosisSteppable.childCell
        self.parentCell = self.parent_cell
        self.childCell = self.child_cell

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

        self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[])

    def update_attributes(self):
        """
        This function is supposed to be reimplemented in the subclass. It is called immediately after cell division
        takes place
        :return:
        """

        self.child_cell.targetVolume = self.parent_cell.targetVolume
        self.child_cell.lambdaVolume = self.parent_cell.lambdaVolume
        self.child_cell.type = self.parent_cell.type

    @deprecated(version='4.0.0', reason="You should use : init_parent_and_child_cells")
    def initParentAndChildCells(self):
        return self.init_parent_and_child_cells()

    def init_parent_and_child_cells(self):
        """
        Initializes self.parentCell and self.childCell to point to respective cell objects after mitosis
         is completed succesfully
        """

        self.parent_cell = self.mitosisSteppable.parentCell
        self.child_cell = self.mitosisSteppable.childCell

        self.parentCell = self.parent_cell
        self.childCell = self.child_cell

    def handle_mitosis_update_attributes(self, mitosis_done):
        """
        Performs actions and bookipping that has to be done after actual cell division happened.
        One of such actions is calling update_attributes function that users typically overload
        :param mitosis_done: {bool} flag indicating if mitosis has been sucessful
        :return: None
        """

        if mitosis_done:
            self.init_parent_and_child_cells()
            try:
                legacy_update_attributes_fcn = getattr(self, 'updateAttributes')
            except AttributeError:
                legacy_update_attributes_fcn = None

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
        return mitosis_done


class MitosisSteppableClustersBase(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.mitosisSteppable = None
        self.parent_cell = None
        self.child_cell = None
        # legacy API
        self.parentCell = None
        self.childCell = None
        self._parent_child_position_flag = None

    def core_init(self, reinitialize_cell_types=True):
        SteppableBasePy.core_init(self, reinitialize_cell_types=reinitialize_cell_types)
        self.mitosisSteppable = CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parent_cell = self.mitosisSteppable.parentCell
        self.child_cell = self.mitosisSteppable.childCell
        self.parentCell = self.parent_cell
        self.childCell = self.child_cell

        # # delayed initialization
        # if self._parent_child_position_flag is not None:
        #     self.set_parent_child_position_flag(flag=self._parent_child_position_flag)

    @deprecated(version='4.0.0', reason="You should use : init_parent_and_child_cells")
    def initParentAndChildCells(self):

        return self.init_parent_and_child_cells()

    def init_parent_and_child_cells(self):
        """
        Initializes self.parent_cell and self.child_cell to point to respective cell objects after mitosis
        is completed successfully
        """

        self.parent_cell = self.mitosisSteppable.parentCell
        self.child_cell = self.mitosisSteppable.childCell

        self.parentCell = self.parent_cell
        self.childCell = self.child_cell

    @deprecated(version='4.0.0', reason="You should use : clone_cluster_attributes")
    def cloneClusterAttributes(self, sourceCellCluster, targetCellCluster, no_clone_key_dict_list=None):

        if no_clone_key_dict_list is None:
            no_clone_key_dict_list = []
        return self.clone_cluster_attributes(source_cell_cluster=sourceCellCluster,
                                             target_cell_cluster=targetCellCluster,
                                             no_clone_key_dict_list=no_clone_key_dict_list)

    def clone_cluster_attributes(self, source_cell_cluster, target_cell_cluster, no_clone_key_dict_list=None):
        """
        Clones attributes for cluster members
        :param source_cell_cluster: {vector of CellG objects} vector (C++) of CellG object representing
        source cluster
        :param target_cell_cluster: {vector of CellG objects} vector (C++) of CellG object representing
        target cluster
        :param no_clone_key_dict_list: {list} list-based specification of attributes that are not supposed
        to be cloned
        :return:
        """

        if no_clone_key_dict_list is None:
            no_clone_key_dict_list = []

        for i in range(source_cell_cluster.size()):
            self.clone_attributes(source_cell=source_cell_cluster[i], target_cell=target_cell_cluster[i],
                                  no_clone_key_dict_list=no_clone_key_dict_list)

    @deprecated(version='4.0.0', reason="You should use : clone_parent_cluster_2_child_cluster")
    def cloneParentCluster2ChildCluster(self):

        return self.clone_parent_cluster_2_child_cluster()

    def clone_parent_cluster_2_child_cluster(self):
        """
        Clones attributes of "parent" cluster to "child" cluster wherre parent and child
        refer to objects that existed before and after mitosis
        these calls seem to be necessary to ensure
        whatever is set in in mitosisSteppable (C++) is reflected in Python
        :return:
        """

        compartment_list_parent = self.inventory.getClusterCells(self.parentCell.clusterId)
        compartment_list_child = self.inventory.getClusterCells(self.childCell.clusterId)

        self.clone_cluster_attributes(source_cell_cluster=compartment_list_parent,
                                      target_cell_cluster=compartment_list_child,
                                      no_clone_key_dict_list=[])

    def update_attributes(self):
        """
        Default implmentation of update attribute function that is called
        immediately after the mitosis happens. This function is supposed
        to be reimplemented in the subclass. Default implementation only copies type
        attribute from parent to child cell
        :return:
        """

        parent_cell = self.mitosisSteppable.parentCell
        child_cell = self.mitosisSteppable.childCell
        compartment_list_child = self.inventory.getClusterCells(child_cell.clusterId)
        compartment_list_parent = self.inventory.getClusterCells(parent_cell.clusterId)
        # compartments in the parent and child clusters arel listed
        # in the same order so attribute changes require simple iteration through compartment list
        for i in range(compartment_list_child.size()):
            compartment_list_child[i].type = compartment_list_parent[i].type

    def handle_mitosis_update_attributes(self, mitosis_done):
        """
        Performs actions and bookipping that has to be done after actual cell division happened.
        One of such actions is calling update_attributes function that users typically overload
        :param mitosis_done: {bool} flag indicating if mitosis has been successful
        :return: None
        """

        if mitosis_done:
            self.init_parent_and_child_cells()
            try:
                legacy_update_attributes_fcn = getattr(self, 'updateAttributes')
            except AttributeError:
                legacy_update_attributes_fcn = None

            if legacy_update_attributes_fcn is not None:
                warnings.warn('"updateAttribute function" is deprecated since 4.0.0. '
                              'Please use "update_attributes" in your'
                              ' mitosis subclass', DeprecationWarning)
                legacy_update_attributes_fcn()
            else:
                self.update_attributes()

    @deprecated(version='4.0.0', reason="You should use : divide_cluster_random_orientation")
    def divideClusterRandomOrientation(self, _clusterId):

        return self.divide_cluster_random_orientation(cluster_id=_clusterId)

    def divide_cluster_random_orientation(self, cluster_id):
        """
        Divides cluster into two daughter clusters along randomly chosen cleavage plane.
        For tracking reasons one daughter cluster is considered a "parent"
        and refers to a set of cell objects that existed before division

        :param cluster_id: {long int} cluster id
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisRandomOrientationCompartments(cluster_id)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cluster_orientation_vector_based")
    def divideClusterOrientationVectorBased(self, _clusterId, _nx, _ny, _nz):

        return self.divide_cluster_orientation_vector_based(cluster_id=_clusterId, nx=_nx, ny=_ny, nz=_nz)

    def divide_cluster_orientation_vector_based(self, cluster_id, nx, ny, nz):
        """
        Divides cluster into two daughter clusters along cleavage plane specified by a normal vector (nx, ny, nz).
        For tracking reasons one daughter cluster is considered a "parent"
        and refers to a set of cell objects that existed before division
        :param cluster_id: {long int} cluster id
        :param nx: {float} 'x' component of vector normal to the cleavage plane
        :param ny: {float} 'y' component of vector normal to the cleavage plane
        :param nz: {float} 'z' component of vector normal to the cleavage plane
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisOrientationVectorBasedCompartments(
            cluster_id, nx, ny, nz)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cluster_along_major_axis")
    def divideClusterAlongMajorAxis(self, cluster_id):

        return self.divide_cluster_along_major_axis(cluster_id=cluster_id)

    def divide_cluster_along_major_axis(self, cluster_id):
        """
        Divides cell into two daughter clusters along cleavage plane parallel to major axis of the cluster.
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a set of cell objects that existed before division

        :param cluster_id: {long int} cluster id
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisAlongMajorAxisCompartments(cluster_id)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)
        return mitosis_done

    @deprecated(version='4.0.0', reason="You should use : divide_cluster_along_minor_axis")
    def divideClusterAlongMinorAxis(self, _clusterId):
        return self.divide_cluster_along_minor_axis(cluster_id=_clusterId)

    def divide_cluster_along_minor_axis(self, cluster_id):
        """
        Divides cell into two daughter clusters along cleavage plane parallel to minor axis of the cluster.
        For tracking reasons one daughter cell is considered a "parent"
        and refers to a set of cell objects that existed before division

        :param cluster_id: {long int} cluster id
        :return: None
        """

        mitosis_done = self.mitosisSteppable.doDirectionalMitosisAlongMinorAxisCompartments(cluster_id)
        self.handle_mitosis_update_attributes(mitosis_done=mitosis_done)

        return mitosis_done


class RunBeforeMCSSteppableBasePy(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.runBeforeMCS = 1


class SecretionBasePy(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.runBeforeMCS = 1
