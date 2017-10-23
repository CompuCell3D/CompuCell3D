from SimulationLauncher import SimulationLauncher
from CLISimulationLauncher import CLISimulationLauncher
from GUISImulationLaucher import GUISimulationLauncher

class SimulationLauncherFactory:
    """
    This class is responsible for creating a concrete object of appropriate Simulation Launcher.
    """
    invocationType = None

    def setInvocationType(self, invocationType):
        self.invocationType = invocationType

    def getSimulationLauncher(self, invocationType = "GUI"):

        simulationLauncherInstance =  None

        if self.invocationType is None:
            self.setInvocationType(invocationType)

        if self.invocationType == "CLI":
            simulationLauncherInstance = self.createCLISimulationLaucher()
        elif self.invocationType == "GUI":
            simulationLauncherInstance = self.createGUISimulationLauncher()

        return simulationLauncherInstance

    def createCLISimulationLaucher(self):
        return CLISimulationLauncher()

    def createGUISimulationLauncher(self):
        return GUISimulationLauncher()