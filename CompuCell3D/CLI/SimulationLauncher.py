import abc
from abc import abstractmethod


class SimulationLauncher:
    """
    This is a abstract class for Simulation Launcher. This class will be newer version of
    CompuCellSetup.
    """
    __metaclass__ = abc.ABCMeta

    def getSteppableRegistry():
        from PySteppables import SteppableRegistry
        steppableRegistry = SteppableRegistry()
        return steppableRegistry
    
    @abstractmethod
    def executeSimulation(self):
        pass