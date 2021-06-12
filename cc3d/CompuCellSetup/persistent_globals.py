import os
import time
from collections import OrderedDict
from os.path import join, exists, basename
from typing import Union
from cc3d.core.SteppableRegistry import SteppableRegistry
from cc3d.core.FieldRegistry import FieldRegistry
import copy
from cc3d.core.utils import mkdir_p
from cc3d.cpp.CompuCell import PyAttributeAdder
from pathlib import Path
from threading import Lock


class PersistentGlobals:
    def __init__(self):
        self.cc3d_xml_2_obj_converter = None
        self.steppable_registry = SteppableRegistry()

        # c++ object reference Simulator.cpp
        self.simulator = None

        #  Simulation Thread - either from the player or from CML
        self.simthread = None

        # hook to player - initialized in the player in the prepareForNewSimulation method
        # this object is not None only when GUI run is requested
        self.view_manager = None

        # an object that writes or reads fields from disk
        self.cml_field_handler = None

        # an object that stores graphics screenshots
        self.screenshot_manager = None

        # variable that tells what type of player mode we have
        self.player_type = None

        self.simulation_initialized = False
        self.simulation_file_name = None
        self.user_stop_simulation_flag = False

        self.__output_dir = None
        self.output_file_core_name = "Step"

        self.__workspace_dir = None

        self.__param_scan_iteration = None

        self.output_frequency = 0
        self.screenshot_output_frequency = 0

        self.restart_snapshot_frequency = 0
        self.restart_multiple_snapshots = 0
        self.restart_manager = None

        # todo - move it elsewhere or come up with a better solution
        # two objects that handle adding addition of python attributes
        self.attribute_adder = None
        self.dict_adder = None

        # object that facilitates fast lookups of XML elements
        self.xml_id_locator = None

        # class - container that stores information about the fields
        self.field_registry = FieldRegistry()

        self.persistent_holder = {}

        self.global_sbml_simulator_options = None
        self.free_floating_sbml_simulators = {}

        # dictionary holding steering parameter objects - used for custom steering panel
        self.steering_param_dict = OrderedDict()
        self.steering_panel_synchronizer = Lock()

        # dictionary holding shared variables between steppables
        self.shared_steppable_vars = {}

        # input and return objects
        self.input_object = None
        self.return_object = None

    def add_steering_panel(self, panel_data: dict):
        """
        Adds steering panel if simulation is run using player
        :param panel_data: dictionary with param scans
        :return:
        """
        if self.view_manager is None:
            return

        steering_panel = self.view_manager.widgetManager.getNewWidget('Steering Panel', panel_data)
        return steering_panel

    def set_output_dir(self, output_dir: str) -> None:
        """
        Sets screenshot dir - usually this is a custom screenshot directory set based on
        command line input

        :param output_dir:
        :return:
        """

        self.__output_dir = output_dir

    def set_workspace_dir(self, workspace_dir: str) -> None:
        """
        Sets screenshot dir - usually this is a custom screenshot directory set based on
        command line input

        :param workspace_dir:
        :return:
        """

        self.__workspace_dir = workspace_dir

    @property
    def parameter_scan_iteration(self):
        """
        returns current parameter scan iteration
        :return:
        """
        return self.__param_scan_iteration

    @parameter_scan_iteration.setter
    def parameter_scan_iteration(self, val):
        self.__param_scan_iteration = val

    @property
    def workspace_dir(self) -> str:
        """
        returns workspace directory
        :return:
        """

        if self.__workspace_dir is None:
            workspace_dir = os.path.join(os.path.expanduser('~'), 'CC3DWorkspace')
            if not exists(workspace_dir):
                Path(workspace_dir).mkdir(parents=True, exist_ok=True)

            return workspace_dir
        else:
            return self.__workspace_dir

    @property
    def timestamp_string(self) -> str:
        """
        returns current timestamp string
        :return:
        """

        current_time = time.localtime()
        str_f_time = time.strftime
        timestamp_str = "_" + str_f_time("%m", current_time) + "_" + str_f_time("%d", current_time) + "_" + str_f_time(
            "%Y", current_time) + "_" + str_f_time("%H", current_time) + "_" + str_f_time("%M",
                                                                                          current_time) + "_" + str_f_time(
            "%S", current_time)

        return timestamp_str

    @property
    def output_directory(self) -> Union[str, None]:
        """
        Retuns screenshot directory - if possible to construct one otherwise returns None
        :return:
        """
        if self.simulation_file_name is None:
            return None
        elif self.__output_dir is not None:
            return self.__output_dir
        else:
            sim_base_name = basename(self.simulation_file_name)
            sim_base_name = sim_base_name.replace('.', '_')
            sim_base_name += self.timestamp_string

            self.__output_dir = join(self.workspace_dir, sim_base_name)

            return self.__output_dir

    def create_output_dir(self):
        """

        :return:
        """

        mkdir_p(self.output_directory)

    def clean(self):
        """

        :return:
        """

    def attach_dictionary_to_cells(self) -> None:
        """
        Utility method that handles addition of
        dictionary to C++ cells.
        :return: None
        """

        if self.attribute_adder is not None:
            return

        if self.simulator is None:
            return

        class DictAdder:
            def __init__(self):
                self.dict_template = {}

            def addAttribute(self):
                """
                function called ducring cell creation in C++
                C++ expects this function to be exactly "addAttribute". Do not refactor
                :return:
                """
                temp_copy = copy.deepcopy(self.dict_template)
                return temp_copy

        self.attribute_adder = PyAttributeAdder()
        # adder.registerRefChecker(getrefcount)
        self.dict_adder = DictAdder()
        self.attribute_adder.registerAdder(self.dict_adder)
        potts = self.simulator.getPotts()
        potts.registerAttributeAdder(self.attribute_adder.getPyAttributeAdderPtr())
