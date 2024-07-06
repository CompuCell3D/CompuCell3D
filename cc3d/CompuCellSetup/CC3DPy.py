import os
import time
from cc3d import CompuCellSetup
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame import CC3DPyGraphicsFrameClient
from typing import Optional


class CC3DPy:
    """
    Defines a set of generic functions for elementary operations when operating CC3D in Python
    """

    @staticmethod
    def init_simulation(cc3d_sim_fname=None,
                        output_frequency=0,
                        screenshot_output_frequency=0,
                        restart_snapshot_frequency=0,
                        restart_multiple_snapshots=False,
                        output_dir=None,
                        output_file_core_name=None,
                        sim_input=None):
        # Always start with fresh persistent globals
        CompuCellSetup.resetGlobals()
        persistent_globals = CompuCellSetup.persistent_globals

        # Populate persistent_globals with basic simulation info
        persistent_globals.simulation_file_name = cc3d_sim_fname
        persistent_globals.output_frequency = output_frequency
        persistent_globals.screenshot_output_frequency = screenshot_output_frequency
        persistent_globals.set_output_dir(output_dir)
        persistent_globals.output_file_core_name = output_file_core_name
        persistent_globals.restart_snapshot_frequency = restart_snapshot_frequency
        persistent_globals.restart_multiple_snapshots = restart_multiple_snapshots
        persistent_globals.input_object = sim_input

    @staticmethod
    def run(cc3d_sim_fname):
        CompuCellSetup.run_cc3d_project(cc3d_sim_fname)

    @staticmethod
    def call_init():
        CompuCellSetup.persistent_globals.steppable_registry.init(CompuCellSetup.persistent_globals.simulator)

        CompuCellSetup.init_lattice_snapshot_objects()
        CompuCellSetup.init_screenshot_manager()

    @staticmethod
    def call_start():
        init_using_restart_snapshot_enabled = CompuCellSetup.persistent_globals.restart_manager.restart_enabled()

        if not init_using_restart_snapshot_enabled:
            CompuCellSetup.persistent_globals.steppable_registry.start()

        CompuCellSetup.persistent_globals.restart_manager.prepare_restarter()

    @staticmethod
    def call_step(current_step: int):
        pg = CompuCellSetup.persistent_globals

        pg.steppable_registry.stepRunBeforeMCSSteppables(current_step)

        compiled_code_begin = time.time()
        pg.simulator.step(current_step)
        CompuCellSetup.check_for_cpp_errors(pg.simulator)
        compiled_code_end = time.time()
        compiled_code_run_time = (compiled_code_end - compiled_code_begin) * 1000

        pg.steppable_registry.step(current_step)

        return compiled_code_run_time

    @staticmethod
    def call_steer(current_step):
        CompuCellSetup.incorporate_script_steering_changes(current_step)
        CompuCellSetup.persistent_globals.simulator.steer()

    @staticmethod
    def store_sim_step_data(current_step):
        CompuCellSetup.persistent_globals.restart_manager.output_restart_files(current_step)
        CompuCellSetup.store_lattice_snapshot(cur_step=current_step)
        CompuCellSetup.store_screenshots(cur_step=current_step)

    @staticmethod
    def store_lattice_snapshot(mcs: int, output_dir_name: str = None, output_file_core_name: str = None) -> bool:
        """
        Store a lattice snapshot on demand

        :param mcs: current step
        :param output_dir_name: override output directory
        :param output_file_core_name: override output file core name
        :return: True on success
        """
        cml_field_handler = CompuCellSetup.persistent_globals.cml_field_handler
        if cml_field_handler:
            cml_field_handler.write_fields(mcs, output_dir_name, output_file_core_name)
            return True
        return False

    @staticmethod
    def check_cc3d():
        try:
            CompuCellSetup.check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)
            result, error_message = True, ""
        except CompuCellSetup.CC3DCPlusPlusError as cc3d_cpp_err:
            result, error_message = False, cc3d_cpp_err.message
        return result, error_message

    @staticmethod
    def get_xml_element(tag: str):
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

    @staticmethod
    def visualization():
        frame = CC3DPyGraphicsFrameClient()
        frame.launch()
        return frame

    @staticmethod
    def get_screenshot_data() -> dict:
        """Gets a copy of the current screenshot data"""

        sm = CompuCellSetup.persistent_globals.screenshot_manager
        if sm is None:
            raise RuntimeError('Screenshot manager not available at this time')

        return sm.generate_screenshot_description_data()

    @staticmethod
    def set_screenshot_data(_data: dict):
        """Sets current screenshot data"""
        sm = CompuCellSetup.persistent_globals.screenshot_manager
        if sm is None:
            raise RuntimeError('Screenshot manager not available at this time')

        sm.read_screenshot_description_data(_data)

    @staticmethod
    def save_screenshot_data(_fp: str = None) -> str:
        """
        Saves screenshot data if available

        :param _fp: absolute file path of saved data; default is standard location if available
        :return: file path if saved, otherwise None
        """

        sm = CompuCellSetup.persistent_globals.screenshot_manager
        if sm is None:
            raise RuntimeError('Screenshot manager not available at this time')

        if _fp is None:
            _fp = sm.get_screenshot_filename()
        if _fp is None:
            raise RuntimeError('Screenshot file name could not be deduced')

        sm.write_screenshot_description_file(_fp)
        return _fp

    @staticmethod
    def load_screenshot_data(_fp: str):
        """
        Loads screenshot data from file

        :param _fp: absolute path of saved data
        :return: true is loaded
        """

        if not os.path.isfile(_fp):
            raise FileNotFoundError('Screenshot data file does not exist at path', _fp)

        sm = CompuCellSetup.persistent_globals.screenshot_manager
        if sm is None:
            raise RuntimeError('Screenshot manager not available at this time')

        sm.read_screenshot_description_file(_fp)
        return True

    @staticmethod
    def starting_screenshot_description_data():
        """Generates the start of screenshot description data"""

        from cc3d.core.GraphicsUtils.ScreenshotManagerCore import ScreenshotManagerCore
        return ScreenshotManagerCore.starting_screenshot_description_data()

    @staticmethod
    def append_screenshot_description_data(root_elem: dict, data_elem: dict, screenshot_uid: Optional[str]=None):
        """
        Append a screenshot data element

        :param root_elem: screenshot data
        :param data_elem: data element for a field
        :return: updated dictionary
        """

        from cc3d.core.GraphicsUtils.ScreenshotManagerCore import ScreenshotManagerCore
        return ScreenshotManagerCore.append_screenshot_description_data(root_elem, data_elem, screenshot_uid)


class CC3DPySim:
    """
    Base class for basic Python-based CC3D simulation instances
    """
    def __init__(self,
                 cc3d_sim_fname=None,
                 output_frequency=0,
                 screenshot_output_frequency=0,
                 restart_snapshot_frequency=0,
                 restart_multiple_snapshots=False,
                 output_dir=None,
                 output_file_core_name=None,
                 sim_input=None):

        self.cc3d_sim_fname = cc3d_sim_fname
        self.output_frequency = output_frequency
        self.screenshot_output_frequency = screenshot_output_frequency
        self.restart_snapshot_frequency = restart_snapshot_frequency
        self.restart_multiple_snapshots = restart_multiple_snapshots
        self.output_dir = output_dir
        self.output_file_core_name = output_file_core_name
        self.simulation_input = sim_input

    def init_simulation(self):
        """
        Initialize CC3D from Python
        :return: None
        """
        CC3DPy.init_simulation(cc3d_sim_fname=self.cc3d_sim_fname,
                               output_frequency=self.output_frequency,
                               screenshot_output_frequency=self.screenshot_output_frequency,
                               restart_snapshot_frequency=self.restart_snapshot_frequency,
                               restart_multiple_snapshots=self.restart_multiple_snapshots,
                               output_dir=self.output_dir,
                               output_file_core_name=self.output_file_core_name,
                               sim_input=self.simulation_input)

    def uninit_simulation(self):
        """
        Uninitialize CC3D from Python
        :return: None
        """
        pass

    def run(self):
        raise NotImplementedError

    def visualization(self):
        return CC3DPy.visualization()

    @staticmethod
    def get_screenshot_data() -> dict:
        """Gets a copy of the current screenshot data, if any"""
        return CC3DPy.get_screenshot_data()

    @staticmethod
    def set_screenshot_data(_data: dict):
        """Sets current screenshot data"""
        return CC3DPy.set_screenshot_data(_data)

    @staticmethod
    def save_screenshot_data(_fp: str = None) -> str:
        """
        Saves screenshot data if available

        :param _fp: absolute file path of saved data; default is standard location if available
        :return: file path if saved, otherwise None
        """
        return CC3DPy.save_screenshot_data(_fp)

    @staticmethod
    def load_screenshot_data(_fp: str):
        """
        Loads screenshot data from file

        :param _fp: absolute path of saved data
        :return: true is loaded
        """
        return CC3DPy.load_screenshot_data(_fp)

    @staticmethod
    def starting_screenshot_description_data():
        """Generates the start of screenshot description data"""
        return CC3DPy.starting_screenshot_description_data()

    @staticmethod
    def append_screenshot_description_data(root_elem: dict, data_elem: dict) -> dict:
        """
        Append a screenshot data element

        :param root_elem: screenshot data
        :param data_elem: data element for a field
        :return: updated dictionary
        """
        CC3DPy.append_screenshot_description_data(root_elem, data_elem)
        # Returning redundant result for service interface
        return root_elem
