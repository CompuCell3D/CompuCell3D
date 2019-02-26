import itertools
from cc3d.core.iterators import *
# from cc3d.CompuCellSetup.simulation_utils import stop_simulation
from cc3d import CompuCellSetup

# try:
#     import cc3d.CompuCellSetup
# except ImportError:
#     pass


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

    def step(self, mcs):
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


        # legacy API
        self.addNewPlotWindow = self.add_new_plot_window
        self.createScalarFieldPy =self.create_scalar_field_py
        self.everyPixelWithSteps = self.every_pixel_with_steps
        self.everyPixel = self.every_pixel


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

        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}




    def stop_simulation(self):
        """
        Stops simulation
        :return:
        """

        CompuCellSetup.stop_simulation()

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

    def add_new_plot_window(self, title, xAxisTitle, yAxisTitle, xScaleType='linear', yScaleType='linear', grid=True, config_options=None):


        if title in self.plot_dict.keys():
            raise RuntimeError('PLOT WINDOW: ' + title + ' already exists. Please choose a different name')

        pW = CompuCellSetup.simulation_player_utils.add_new_plot_window(title, xAxisTitle, yAxisTitle, xScaleType, yScaleType, grid, config_options=config_options)
        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}


        return pW



    def create_scalar_field_py(self, fieldName):
        """
        Created extra visualization field
        :param fieldName: {str}
        :return:
        """
        return CompuCellSetup.simulation_player_utils.create_scalar_field_py(fieldName)
        # return CompuCellSetup.simulation_player_utils.create_scalar_field_py(self.dim, fieldName)


    def every_pixel_with_steps(self, step_x, step_y, step_z):
        """
        Helper function called by every_pixel method. See documentation of every_pixel for details
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """
        for x in range(0, self.dim.x, step_x):
            for y in range(0, self.dim.y, step_y):
                for z in range(0, self.dim.z, step_z):
                    yield x, y, z

    def every_pixel(self, step_x=1, step_y=1, step_z=1):
        """
        Returns iterator that walks through pixels of the lattixe. Step variables
        determine if we walk through every pixel - step=1 in this case
        or if we jump step variables then are > 1
        :param step_x:
        :param step_y:
        :param step_z:
        :return:
        """
        if step_x == 1 and step_y == 1 and step_z == 1:

            return itertools.product(range(self.dim.x), range(self.dim.y), range(self.dim.z))
        else:
            return self.every_pixel_with_steps(step_x, step_y, step_z)


