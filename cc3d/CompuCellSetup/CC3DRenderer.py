import os
import time
from typing import Union

from cc3d.cpp import CompuCell
from cc3d.core.BasicSimulationData import BasicSimulationData
from cc3d.core.CMLResultsReader import CMLResultReader
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawerCC3DPy
from cc3d.core.GraphicsUtils.ScreenshotManagerCore import ScreenshotManagerCC3DPy
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object

from .utils import SCREENSHOT_SUBDIR, SCREENSHOT_SPEC
# For convenience and backward compatibility:
from .utils import standard_lds_file, standard_screenshot_file


class CC3DRenderer:
    def __init__(self, lds_file: str, screenshot_spec: Union[dict, str] = None, output_dir: str = None):
        """
        Renders CC3D spatial data from Python
        :param lds_file: absolute path to lattice data summary file
        :param screenshot_spec: absolute path to screenshot specification json file or screenshot data
        :param output_dir: absolute path to output directory containing simulation data
        """
        lds_file = os.path.abspath(lds_file)
        assert os.path.isfile(lds_file)
        self.lds_file = lds_file
        self.lds_dir = os.path.dirname(self.lds_file)

        if output_dir is None:
            self.output_dir = os.path.dirname(self.lds_dir)
        else:
            output_dir = os.path.abspath(output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.output_dir = output_dir

        # Function for doing manipulations of screenshot manager and generic drawer at specified steps
        # Signature of manipulator must be None(scm: ScreenshotManagerCC3DPy, gd: GenericDrawerCC3DPy, mcs: int)
        self.render_manipulator = None

        self.screenshot_spec = screenshot_spec
        if self.screenshot_spec is None:
            self.screenshot_spec = os.path.join(self.lds_dir, SCREENSHOT_SUBDIR, SCREENSHOT_SPEC)
        if not isinstance(self.screenshot_spec, dict) and not os.path.isfile(self.screenshot_spec):
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_DEBUG, f'Warning: no current screenshot specification')

        # Minimal peripheral initializations

        self.cml_results_reader = CMLResultReader(file_name=self.lds_file)
        self.cml_results_reader.extract_lattice_description_info(self.lds_file)
        self.lds_file_list = self.cml_results_reader.ldsFileList
        self.available_mcs = [self.cml_results_reader.extract_mcs_number_from_file_name(x) for x in self.lds_file_list]

        self.generic_drawer = GenericDrawerCC3DPy()

        self.screenshot_manager = ScreenshotManagerCC3DPy(screenshot_dir_name=self.output_dir)

    def initialize(self):
        """
        Sets up peripherals
        :return: None
        """
        CompuCell.CC3DLogger.get().log(CompuCell.LOG_INFORMATION, f'Initializing from {self.lds_file}')

        self.generic_drawer.set_field_extractor(self.cml_results_reader.field_extractor)

        self.screenshot_manager.gd = self.generic_drawer
        self.screenshot_manager.bsd = BasicSimulationData()
        self.screenshot_manager.bsd.fieldDim = self.cml_results_reader.fieldDim
        self.screenshot_manager.bsd.numberOfSteps = self.cml_results_reader.numberOfSteps
        if self.screenshot_spec is not None:
            if isinstance(self.screenshot_spec, str):
                self.screenshot_manager.read_screenshot_description_file(self.screenshot_spec)
            else:
                self.screenshot_manager.read_screenshot_description_data(self.screenshot_spec)

    def set_screenshot_spec(self, _screenshot_spec: Union[dict, str]):
        """
        Sets current json screenshot specification
        All subsequent rendering will be done according to these specifications
        :param _screenshot_spec: path to json screenshot specification (can be generated in Player) or screenshot data
        :return: None
        """
        if isinstance(_screenshot_spec, str):
            _screenshot_spec = os.path.abspath(_screenshot_spec)
            if os.path.isfile(_screenshot_spec):
                self.screenshot_spec = _screenshot_spec
            else:
                raise FileExistsError(f'File not found: {_screenshot_spec}')
            self.screenshot_manager.read_screenshot_description_file(self.screenshot_spec)
        else:
            self.screenshot_spec = _screenshot_spec
            self.screenshot_manager.read_screenshot_description_data(self.screenshot_spec)

    def set_render_manipulator(self, manipulator):
        """
        Sets the function for doing manipulations of screenshot manager and generic drawer at specified steps
        :param manipulator: function with signature None(scm: ScreenshotManagerCC3DPy, gd: GenericDrawerCC3DPy, mcs: int)
        :return: None
        """
        self.render_manipulator = manipulator

    def step_from_vtk(self, vtk_file_name: str) -> int:
        """
        Returns simulation step from vtk file
        :param vtk_file_name: absolute path of vtk file
        :return: {int} simulation step of vtk file
        """
        if not os.path.isfile(vtk_file_name):
            vtk_file_name = os.path.abspath(vtk_file_name)
            raise FileExistsError(f'File not found: {vtk_file_name}')
        return self.cml_results_reader.extract_mcs_number_from_file_name(vtk_file_name)

    def render_step(self, mcs: int) -> bool:
        """
        Executes screenshot rendering for a simulation step
        :param mcs: simulation step
        :return: {bool} True if rendering succeeded
        """
        if mcs not in self.available_mcs:
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_DEBUG, f'Step {mcs} not available')
            return False

        file_number = self.available_mcs.index(mcs)
        self.cml_results_reader.read_simulation_data(file_number)
        sim_data_int_addr = extract_address_int_from_vtk_object(self.cml_results_reader.simulationData)
        self.generic_drawer.field_extractor.setSimulationData(sim_data_int_addr)
        if self.render_manipulator is not None:
            self.render_manipulator(self.screenshot_manager, self.generic_drawer, mcs)
        return True

    def render_vtk(self, vtk_file_name) -> bool:
        """
        Executes screenshot rendering for a vtk file
        :param vtk_file_name: vtk file name
        :return: {bool} True if rendering succeeded
        """
        return self.render_step(self.step_from_vtk(vtk_file_name))

    def export_step(self, mcs: int) -> bool:
        """
        Executes screenshot rendering and write to disk for a simulation step
        :param mcs: simulation step
        :return: {bool} True if rendering succeeded
        """
        if not self.render_step(mcs):
            return False
        self.screenshot_manager.output_screenshots(mcs)
        return True

    def export_vtk(self, vtk_file_name) -> bool:
        """
        Executes screenshot rendering and write to disk for a vtk file
        :param vtk_file_name: vtk file name
        :return: {bool} True if rendering succeeded
        """
        return self.export_step(self.step_from_vtk(vtk_file_name))

    def export_all(self) -> list:
        """
        Executes screenshot rendering and write to disk for all loaded simulation steps
        :return: {list(bool)} True if rendering succeeded
        """
        return [self.export_step(mcs=mcs) for mcs in self.available_mcs]


class CC3DBatchRenderer:
    def __init__(self, lds_files: list, screenshot_spec, output_dirs: list, manipulators=None):
        """
        Executes batch rendering in parallel; see CC3DRenderer for details of inputs
        :param lds_files: {list} list of lattice data summary files
        :param screenshot_spec: screenshot specification json; can be per lds_files, or uniformly applied
        :param output_dirs: {list} list of output directories
        :param manipulators: rendering manipulator; can be per lds_files, or uniformly applied
        """
        # Apply uniform inputs
        if isinstance(screenshot_spec, str) or isinstance(screenshot_spec, dict):
            screenshot_spec = [screenshot_spec] * len(lds_files)

        self.manipulators = manipulators
        if manipulators is not None and not isinstance(manipulators, list):
            self.manipulators = [manipulators] * len(lds_files)

        # Validate inputs
        assert len(lds_files) == len(output_dirs)
        assert not any([not os.path.isfile(f) for f in lds_files])
        if isinstance(screenshot_spec[0], str):
            assert not any([not os.path.isfile(f) for f in screenshot_spec])
        assert not any([not os.path.isdir(d) for d in output_dirs])

        self.lds_files = lds_files
        self.output_dirs = output_dirs
        self.screenshot_spec = screenshot_spec

    @staticmethod
    def _put_with_wait(_tasks, _job):
        while _tasks.full():
            time.sleep(1)
        _tasks.put(_job)

    def export_all(self, num_workers: int = 1):
        """
        Render and export spatial data in parallel
        :param num_workers: {int} number of threads to do batch rendering
        :return: None
        """
        assert num_workers > 0

        import multiprocessing
        from cc3d.CompuCellSetup.CC3DCaller import CC3DCallerWorker

        # Start workers
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [CC3DCallerWorker(tasks, results) for _ in range(num_workers)]
        [w.start() for w in workers]

        # Enqueue jobs
        for r in range(len(self.lds_files)):
            self._put_with_wait(tasks, _RenderDataJob(self.lds_files[r],
                                                      self.screenshot_spec[r],
                                                      self.output_dirs[r],
                                                      self.manipulators[r] if self.manipulators is not None else None))

        # Add a stop task for each of worker
        [self._put_with_wait(tasks, None) for _ in workers]

        tasks.join()


class _RenderDataJob:
    def __init__(self, _lds_file: str, _screenshot_spec: Union[dict, str], _output_dir: str, _manipulator):
        assert os.path.isfile(_lds_file)
        if isinstance(_screenshot_spec, str):
            assert os.path.isfile(_screenshot_spec)
        assert os.path.isdir(_output_dir)

        self._lds_file = _lds_file
        self._screenshot_spec = _screenshot_spec
        self._output_dir = _output_dir
        self._manipulator = _manipulator

    def run(self):
        try:
            r = CC3DRenderer(lds_file=self._lds_file,
                             screenshot_spec=self._screenshot_spec,
                             output_dir=self._output_dir)
            r.initialize()
            if self._manipulator is not None:
                r.set_render_manipulator(self._manipulator)
            r.export_all()
            return True
        except Exception:
            return False
