"""
CC3D core implementation of simservice PySimService
"""
import os
import time
from typing import Any, List, Optional

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.CC3DPy import CC3DPy, CC3DPySim
from cc3d.core.SteppablePy import SteppablePy
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame import CC3DPyGraphicsFrameClient, CC3DPyGraphicsFrameClientBase
from cc3d.core.GraphicsUtils.JupyterGraphicsFrameWidget import JupyterGraphicsFrameClient
from cc3d.core.RollbackImporter import RollbackImporter

from simservice.PySimService import PySimService, SimStatus
from .SimulationServiceThread import SimulationServiceThread


def _in_jupyter():
    try:
        get_ipython
        return True
    except NameError:
        return False


class GraphicsFrameContainer:
    """Container for a graphics frame and associated data"""

    def __init__(self,
                 frame: CC3DPyGraphicsFrameClientBase,
                 plot_freq: int,
                 blocking: bool):

        self.frame = frame
        self.plot_freq = plot_freq
        self.blocking = blocking


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
        self._corespecs_queue = list()

        self._graphics_frames: List[GraphicsFrameContainer] = []
        """Graphics frames synchronized with this service"""

    def __del__(self):
        self.uninit_simulation()
        self.close_frames()

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
            if not isinstance(steppable, SteppablePy):
                steppable = steppable(frequency=frequency)
            CompuCellSetup.register_steppable(steppable=steppable)
        for _ in range(len(self._corespecs_queue)):
            CompuCellSetup.register_specs(self._corespecs_queue.pop(0))

        # Load model specs from file if specified
        if self.cc3d_sim_fname is not None:
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

    def register_specs(self, *args):
        """
        Register Python core spec(s) with CC3D
        This should be performed before initializing simulation, and cannot be mixed with CC3DML
        Core specs will be registered during call to run
        :param args: one or more core specs
        :return: None
        """
        self._corespecs_queue.extend(*args)

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

        for frame_c in self._graphics_frames:
            if self._current_step % frame_c.plot_freq == 0:
                frame_c.frame.draw(blocking=frame_c.blocking)

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
        self.close_frames()

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

    def store_lattice_snapshot(self, output_dir_name: str = None, output_file_core_name: str = None):
        """
        Store a lattice snapshot on demand
        :param output_dir_name: override output directory
        :param output_file_core_name: override output file core name
        :return: True on success
        """
        return CC3DPy.store_lattice_snapshot(self.current_step, output_dir_name, output_file_core_name)

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

    def visualization(self):
        return self.visualize()

    def _load_viz_frame(self,
                        frame: CC3DPyGraphicsFrameClientBase,
                        plot_freq: int,
                        blocking: bool,
                        timeout: float,
                        drawing_style: str):

        def _on_frame_close():
            for i in range(len(self._graphics_frames)):
                if self._graphics_frames[i].frame is frame:
                    self._graphics_frames.pop(i)
                    return

        frame.add_callback_close(_on_frame_close)
        self._graphics_frames.append(GraphicsFrameContainer(frame=frame,
                                                            plot_freq=plot_freq,
                                                            blocking=blocking))
        return_obj = frame.launch(timeout=timeout)

        if drawing_style is not None:
            frame.set_drawing_style(_style=drawing_style)

        return return_obj

    def visualize(self,
                  plot_freq: int = 1,
                  name: str = None,
                  fps: int = 60,
                  config_fp: str = None,
                  blocking: bool = True,
                  timeout: float = None,
                  drawing_style: str = None) -> Optional[Any]:
        """
        Generate a synchronized graphics visualization frame.

        The returned frame is operated by the service, including closing it on service finish.

        :param plot_freq: frequency of plot updates
        :type plot_freq: int
        :param name: name of frame
        :type name: str
        :param fps: frames per second
        :type fps: int
        :param config_fp: filepath to saved frame configuration settings dictionary
        :type config_fp: str
        :param blocking: flag to block when updating renders; should only be disabled when data will not change
        :type blocking: bool
        :param timeout: timeout for launching the frame, in seconds
        :type timeout: float
        :param drawing_style: style of drawing ('2D' or '3D')
        :type drawing_style: str
        :return: visualization frame interface object
        :rtype: Any or None
        """

        if _in_jupyter():
            frame = JupyterGraphicsFrameClient(config_fp=config_fp)
        else:
            frame = CC3DPyGraphicsFrameClient(name=name, fps=fps, config_fp=config_fp)

        return self._load_viz_frame(frame=frame,
                                    plot_freq=plot_freq,
                                    blocking=blocking,
                                    timeout=timeout,
                                    drawing_style=drawing_style)

    def close_frames(self):
        graphics_frames = [frame_c for frame_c in self._graphics_frames]
        for frame_c in graphics_frames:
            frame_c.frame.close()
        self._graphics_frames.clear()

    def jupyter_run_button(self, update_rate: float = 1E-21):
        """
        Get a button that toggles stepping.

        :param update_rate: rate at which the service step method is called, in 1/s;
            slows execution, but allows frame interactions during stepping
        :return: toggle button
        :rtype: ipywidgets.ToggleButton
        :raise RuntimeError: when called outside of a Jupyter environment
        """
        if not _in_jupyter():
            raise RuntimeError('This method is reserved for Jupyter environments')

        import asyncio
        import ipywidgets

        _running = False

        async def _run():
            while True:
                if _running:
                    self.step()
                await asyncio.sleep(update_rate)

        asyncio.ensure_future(_run())

        def _run_cb(change):
            if change['name'] == 'value':
                nonlocal _running
                _running = change.new

        _run_button = ipywidgets.ToggleButton(value=False, description='Run Simulation')
        _run_button.observe(_run_cb, names='value')
        return _run_button
