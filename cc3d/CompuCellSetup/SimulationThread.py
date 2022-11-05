from cc3d.core.enums import SimType


class SimulationThread:
    """
    A minimal class for hooking a custom simulation controller into the CC3D core

    For reference to hooking into the CC3D core when designing a derived class,
    see cc3d.CompuCellSetup.simulation_setup.py
    """

    sim_type = SimType.THREADED

    def __init__(self):
        pass

    def inject(self):
        """
        Inject this into the CC3D core

        :return: None
        """
        from cc3d.CompuCellSetup import persistent_globals
        persistent_globals.simthread = self
        persistent_globals.sim_type = self.sim_type

    @staticmethod
    def main_loop():
        """
        Sequence to control what occurs during a run of a simulation

        If this is not overloaded, then CC3D will function out-of-the-box

        If CC3D detects this in persistent_globals.simthread, then it will use what's returned by this function to
            perform the simulation

        The returned function must have the form
            def f(CompuCell.Simulator, SimulationThread, CompuCell.SteppableRegistry) -> None

        Note that persistent_globals.player_type is an optional global variable to use for conditionalizing which
            function is returned here

        See cc3d.CompuCellSetup.simulation_setup.main_loop for an example of how to set up an automated simulation run

        :return: None
        """
        from cc3d.CompuCellSetup import main_loop
        return main_loop

    def postStartInit(self):
        """
        Signal to SimulationThread to prepare for an upcoming simulation

        Called just after concentration field names have been validated and output directory has been set

        If not restarting, then start() is called on core Simulator just before this call

        See cc3d.CompuCellSetup.simulation_setup.extra_init_simulation_objects
            If not using extra_init_simulation_objects to perform extra inits, then this is not necessarily called

        :return: None
        """
        pass

    def waitForPlayerTaskToFinish(self):
        """
        Blocking call for SimulationThread to complete initialization

        Called just after SimulationThread.postStartInit()

        See cc3d.CompuCellSetup.simulation_setup.extra_init_simulation_objects
            If not using extra_init_simulation_objects to perform extra inits, then this is not necessarily called

        :return: None
        """
        pass

    def emitErrorFormatted(self, _error_message):
        """
        Handle errors in run

        See cc3d.CompuCellSetup.sim_runner.handle_error

        :param _error_message:
        :return:
        """
        pass

    def sendStopSimulationRequest(self):
        """
        Called to send request to stop simulation

        :return: None
        """
        pass

    def simulationFinishedPostEvent(self, _flag=True):
        """
        Called when a simulation has finished

        :param _flag:
        :type _flag:
        :return:
        :rtype:
        """
        pass

    def add_visualization_field(self, field_name, field_type):
        """
        Called when adding a visualization field

        :param field_name: name of field
        :type field_name: str
        :param field_type: type of field
        :type field_type: str
        :return: None
        """
        pass

    def get_field_storage(self):
        """
        Returns field storage

        :return: field storage
        :rtype: cc3d.cpp.PlayerPython.FieldStorage
        """
        pass
