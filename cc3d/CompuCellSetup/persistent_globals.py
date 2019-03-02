import os
import time
from os.path import join, exists, basename

from cc3d.core.SteppableRegistry import SteppableRegistry
from cc3d.core.FieldRegistry import FieldRegistry

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

        self.simulation_initialized = False
        self.simulation_file_name = None
        self.user_stop_simulation_flag = False

        # class - container that stores information about the fields
        self.field_registry = FieldRegistry()

    @property
    def workspace_dir(self)->str:
        """

        :return:
        """
        workspace_dir = os.path.join(os.path.expanduser('~'), 'CC3DWorkspace')
        if not exists(workspace_dir):
            os.mkdir(workspace_dir)

        return workspace_dir

    @property
    def timestamp_string(self):
        current_time = time.localtime()
        str_f_time = time.strftime
        timestamp_str = "_" + str_f_time("%m", current_time) + "_" + str_f_time("%d",current_time) + "_" + str_f_time(
            "%Y", current_time) + "_" + str_f_time("%H", current_time) + "_" + str_f_time("%M",current_time) + "_" + str_f_time(
            "%S", current_time)

        return timestamp_str

    @property
    def screenshot_directory(self):
        if self.simulation_file_name is None:
            return None
        else:
            sim_base_name = basename(self.simulation_file_name)
            sim_base_name = sim_base_name.replace('.', '_')
            sim_base_name += self.timestamp_string

            return join(self.workspace_dir,sim_base_name)

    def clean(self):
        """

        :return:
        """
