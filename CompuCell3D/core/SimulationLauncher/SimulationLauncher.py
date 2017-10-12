import abc
from abc import abstractmethod


class SimulationLauncher:
    """
    This is a abstract class for Simulation Launcher. This class will be newer version of
    CompuCellSetup.
    """
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def executeSimulation(self):
        pass