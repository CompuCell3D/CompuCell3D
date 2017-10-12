from SimulationLauncher import SimulationLauncher
from CLISimulationLauncher import CLISimulationLauncher

class SimulationLauncherFactory:
    """
    This class is responsible for creating a concrete object of appropriate Simulation Launcher.
    """

    def getSimulationLauncher(self, invocationType = "CLI"):
        pass