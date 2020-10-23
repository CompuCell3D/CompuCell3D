import os
import time

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.CC3DPy import CC3DPy, CC3DPySim
from cc3d.core.RollbackImporter import RollbackImporter

from ...PySimService import PySimService, SimStatus
from .SimulationServiceThread import SimulationServiceThread


class CC3DSimService(CC3DPySim, PySimService):
    def __init__(self,
                 cc3d_sim_fname=None,
                 output_frequency=0,
                 screenshot_output_frequency=0,
                 restart_snapshot_frequency=0,
                 restart_multiple_snapshots=False,
                 output_dir=None,
                 output_file_core_name=None,
                 sim_input=None,
                 sim_name: str = ''):
        CC3DPySim.__init__(self, cc3d_sim_fname=cc3d_sim_fname,
                           output_frequency=output_frequency,
                           screenshot_output_frequency=screenshot_output_frequency,
                           restart_snapshot_frequency=restart_snapshot_frequency,
                           restart_multiple_snapshots=restart_multiple_snapshots,
                           output_dir=output_dir,
                           output_file_core_name=output_file_core_name,
                           sim_input=sim_input)
        PySimService.__init__(self, sim_name=sim_name)

        # Simulation thread
        self.simthread = SimulationServiceThread()

        # Performance trackers
        self.compiled_code_run_time = 0.0
        self.total_run_time = 0.0

        # Queue for steppables to be registered that were instantiated outside environment
        # This is needed to safely register steppables through multiple instantiations of CC3D
        # List elements are tuples of the steppable class and frequency
        self._steppable_queue = list()

    def __del__(self):
        self.uninit_simulation()

    def _run(self):
        """
        Called by run; all prep for the underlying simulation is complete after this call!
        :return: None
        """

        self.init_simulation()

        # Inject ourselves into the core
        self.simthread.inject()

        # register steppables in queue
        for _ in range(len(self._steppable_queue)):
            steppable, frequency = self._steppable_queue.pop(0)
            CompuCellSetup.register_steppable(steppable=steppable(frequency=frequency))

        # Load model specs from file if specified
        if self.cc3d_sim_fname is not None:
            assert self.cc3d_sim_fname is not None, "No simulation file set"
            assert os.path.isfile(self.cc3d_sim_fname), f"Could not find simulation file: {self.cc3d_sim_fname}"
            rollback_importer = RollbackImporter()
            CompuCellSetup.run_cc3d_project(self.cc3d_sim_fname)
            rollback_importer.uninstall()

        # Call startup if loaded simulation file didn't already do it
        if CompuCellSetup.persistent_globals.simulator is None:
            CompuCellSetup.run()

    def run(self):
        PySimService.run(self)

    @property
    def sim_input(self):
        return CompuCellSetup.persistent_globals.input_object

    @sim_input.getter
    def sim_input(self):
        return CompuCellSetup.persistent_globals.input_object

    @sim_input.setter
    def sim_input(self, _input):
        CompuCellSetup.persistent_globals.input_object = _input

    @property
    def sim_output(self):
        return CompuCellSetup.persistent_globals.return_object

    @sim_output.getter
    def sim_output(self):
        return CompuCellSetup.persistent_globals.return_object

    @sim_output.setter
    def sim_output(self, _):
        raise AttributeError("CC3D simulation output is set within CC3D")

    def register_steppable(self, steppable, frequency=1):
        """
        Register Python steppable with CC3D
        This should be performed before initializing simulation
        Steppables will be registered during call to run
        :param steppable: steppable
        :param frequency: frequency of calls to steppable.step()
        :return: None
        """
        self._steppable_queue.append((steppable, frequency))

    def _check_cc3d(self):
        result, error_message = CC3DPy.check_cc3d()
        if len(error_message) > 0:
            self._error_message = error_message
        return result

    def _init(self) -> bool:
        """
        Called by init; initialize underlying simulation
        :return: {bool} True if started; False if further start calls are required
        """
        CC3DPy.call_init()

        return True

    def _start(self) -> bool:
        """
        Called by start; after simulation and before stepping
        Should set self.beginning_step to first first step of current_step counter
        :return: {bool} True if started; False if further start calls are required
        """
        CC3DPy.call_start()

        self.beginning_step = CompuCellSetup.persistent_globals.restart_manager.get_restart_step()

        return True

    def _step(self) -> bool:
        """
        Called by step; execute a step of the underlying simulation
        :return: {bool} True if successful, False if something failed
        """

        total_run_time_begin = time.time()

        try:
            self.compiled_code_run_time += CC3DPy.call_step(self._current_step)
        except CompuCellSetup.CC3DCPlusPlusError as cc3d_cpp_err:
            self._error_message = cc3d_cpp_err.message
            return False

        CC3DPy.store_sim_step_data(self._current_step)
        if not self._check_cc3d():
            self.status = SimStatus.SIM_FAILED
            return False

        try:
            CC3DPy.call_steer(self._current_step)  # Need an interface to write XML-based data in Python
            self._check_cc3d()  # Test if this is necessary
        except CompuCellSetup.CC3DCPlusPlusError as cc3d_cpp_err:
            self._error_message = cc3d_cpp_err.message
            self.status = SimStatus.SIM_FAILED
            return False

        total_run_time_end = time.time()
        self.total_run_time += (total_run_time_end - total_run_time_begin) * 1000

        return True

    def _finish(self):
        """
        Called by finish; execute underlying simulation finish
        :return: None
        """
        steppable_registry = CompuCellSetup.persistent_globals.steppable_registry
        steppable_registry.finish()

    def _stop(self, terminate_sim: bool = True):
        """
        Called by stop; execute underlying simulation stop
        :param terminate_sim: {bool} Terminates simulation if True
        :return: None
        """
        steppable_registry = CompuCellSetup.persistent_globals.steppable_registry
        steppable_registry.on_stop()

    def steer(self) -> bool:
        """
        Execute steering; must be called before ad-hoc changes to simulation inputs show up in CC3D
        :return: {bool} True if OK, False if something went wrong
        """
        CompuCellSetup.persistent_globals.input_object = self.sim_input
        try:
            CC3DPy.call_steer(self._current_step)
            self._check_cc3d()  # Test if this is necessary
            return True
        except CompuCellSetup.CC3DCPlusPlusError as cc3d_cpp_err:
            self._error_message = cc3d_cpp_err.message
            return False

    @property
    def profiler_report(self) -> str:
        steppable_registry = CompuCellSetup.persistent_globals.steppable_registry
        if steppable_registry is None:
            return ""
        try:
            return CompuCellSetup.generate_profiling_report(
                py_steppable_profiler_report=CompuCellSetup.persistent_globals.steppable_registry.get_profiler_report(),
                compiled_code_run_time=self.compiled_code_run_time,
                total_run_time=self.total_run_time)
        except ZeroDivisionError:
            return ""
