from enum import Enum


class SimStatus(Enum):
    SIM_REGISTERED = 0
    SIM_LOADED = 1
    SIM_INITIALIZED = 2
    SIM_STARTED = 3
    SIM_RUNNING = 4
    SIM_STOPPED = 5
    SIM_FINISHED = 6
    SIM_FAILED = -1


class PySimService:
    """
    Client-side interface for simulation service processes
    Implementations should derive a wrap for an underlying service from this class

    Basic usage is
        sim_service = PySimService()
        sim_service.run()
        sim_service.init()
        sim_service.start()
        for s in range(S):
            sim_service.step()
        sim_service.finish()

    Status reporting is as follows
        sim_service = PySimService()          : sim_service.status -> SimStatus.REGISTERED
        sim_service.run()                     : sim_service.status -> SimStatus.SIM_LOADED
        sim_service.init()                    : sim_service.status -> SimStatus.SIM_INITIALIZED
        sim_service.start()                   : sim_service.status -> SimStatus.SIM_STARTED
        sim_service.step()                    : sim_service.status -> SimStatus.SIM_RUNNING
        sim_service.finish()                  : sim_service.status -> SimStatus.SIM_FINISHED
        sim_service.stop()                    : sim_service.status -> SimStatus.SIM_STOPPED
        sim_service.stop(terminate_sim=False) : sim_service.status -> SimStatus.SIM_FINISHED

    """
    def __init__(self, sim_name: str = '', *args, **kwargs):

        # Simulation details
        self._sim_name = sim_name
        self.beginning_step = -1
        self._current_step = -1

        # In case of failure
        self._error_message = None

        self.status = SimStatus.SIM_REGISTERED

        # Hook for control in parallel applications
        self._inside_run = self.inside_run

    @property
    def current_step(self):
        return self._current_step

    @property
    def error_message(self):
        return self._error_message

    def run(self):
        """
        Initialize underlying simulation; all prep for the underlying simulation is complete after this call
        :return: name and reference of this service instance
        """

        self._run()

        self.status = SimStatus.SIM_LOADED

        self._inside_run(self)

        return {'name': self._sim_name, 'sim': self}

    @staticmethod
    def inside_run(self):
        """
        Called inside run; this supports parallel applications
        If running a service in parallel, overload this function or set it via set_inside_run with what to do when
        this service acts without further control from the calling process
        :return: None
        """
        pass

    def set_inside_run(self, _inside_run_func):
        """
        Set inside run function; signature must be def f(SimService: cc3d_sim) -> None:
        Usage is simply cc3d_sim.set_inside_run(f)
        :param _inside_run_func: inside run function
        :return: None
        """
        self._inside_run = _inside_run_func

    def set_sim_name(self, _sim_name: str):
        """
        Set simulation name after instantiation
        :param _sim_name:
        :return: None
        """
        self._sim_name = _sim_name

    def init(self) -> bool:
        """
        Initialize underlying simulation
        :return: {bool} True if started; False if further start calls are required
        """
        init_status: bool = self._init()

        if init_status:
            self.status = SimStatus.SIM_INITIALIZED

        return init_status

    def start(self) -> bool:
        """
        After simulation and before stepping
        :return: {bool} True if started; False if further start calls are required
        """
        start_status: bool = self._start()

        if start_status:
            self._current_step = self.beginning_step
            self.status = SimStatus.SIM_STARTED

        return start_status

    def step(self) -> bool:
        """
        Execute a step of the underlying simulation
        :return: {bool} True if successful, False if something failed
        """

        step_status = self._step()

        if step_status:
            self.status = SimStatus.SIM_RUNNING
            self._current_step += 1

        return step_status

    def finish(self):
        """
        Execute underlying simulation finish
        :return: None
        """
        self._finish()

        self.status = SimStatus.SIM_FINISHED

    def stop(self, terminate_sim: bool = True):
        """
        Execute underlying stop
        :param terminate_sim: {bool} Terminates simulation if True
        :return: None
        """
        self._stop(terminate_sim=terminate_sim)

        if terminate_sim:
            self.status = SimStatus.SIM_FINISHED
        else:
            self.status = SimStatus.SIM_STOPPED

    def _run(self):
        """
        Called by run; all prep for the underlying simulation is complete after this call!
        :return: None
        """
        raise NotImplementedError

    def _init(self) -> bool:
        """
        Called by init; initialize underlying simulation
        :return: {bool} True if started; False if further start calls are required
        """
        raise NotImplementedError

    def _start(self) -> bool:
        """
        Called by start; after simulation and before stepping
        Should set self.beginning_step to first first step of current_step counter
        :return: {bool} True if started; False if further start calls are required
        """
        raise NotImplementedError

    def _step(self) -> bool:
        """
        Called by step; execute a step of the underlying simulation
        :return: {bool} True if successful, False if something failed
        """
        raise NotImplementedError

    def _finish(self):
        """
        Called by finish; execute underlying simulation finish
        :return: None
        """
        raise NotImplementedError

    def _stop(self, terminate_sim: bool = True):
        """
        Called by stop; execute underlying simulation stop
        :param terminate_sim: {bool} Terminates simulation if True
        :return: None
        """
        pass

    def steer(self) -> bool:
        """
        Execute steering; calling signal for ad-hoc changes to service and underlying simulation data
        :return: {bool} True if OK, False if something went wrong
        """
        return True

    @property
    def profiler_report(self) -> str:
        """
        Return on-demand profiling information about simulation service
        :return: {str} profiling information
        """
        return ""
