from cc3d.core.iterators import *
class SteppablePy:
    def __init__(self):
        self.runBeforeMCS = 0

    def core_init(self):
        """

        :return:
        """

    def start(self):
        """

        :return:
        """

    def step(self, _mcs):
        """

        :param _mcs:
        :return:
        """

    def finish(self):
        """

        :return:
        """

    def cleanup(self):
        """

        :return:
        """

class SteppableBasePy(SteppablePy):

    (CC3D_FORMAT, TUPLE_FORMAT) = range(0, 2)

    def __init__(self, simulator=None, frequency=1):
        SteppablePy.__init__(self)
        # SBMLSolverHelper.__init__(self)
        self.frequency = frequency
        self.simulator = simulator

    def core_init(self):

        self.potts = self.simulator.getPotts()
        self.cellField = self.potts.getCellFieldG()
        self.dim = self.cellField.getDim()
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.clusterInventory = self.inventory.getClusterInventory()
        self.cellList = CellList(self.inventory)
        self.cellListByType = CellListByType(self.inventory)
        self.clusterList = ClusterList(self.inventory)
        self.clusters = Clusters(self.inventory)
        self.mcs = -1





    def init(self, _simulator):
        """

        :param _simulator:
        :return:
        """

    def add_steering_panel(self):
        """

        :return:
        """

    def process_steering_panel_data_wrapper(self):
        """

        :return:
        """

    def set_steering_param_dirty(self, flag=False):
        """

        :return:
        """
