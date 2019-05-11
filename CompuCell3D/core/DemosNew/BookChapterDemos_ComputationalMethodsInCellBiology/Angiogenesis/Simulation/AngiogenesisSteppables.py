from cc3d.core.PySteppables import *


class AngiogenesisStetppable(SteppableBasePy):

    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        function called before simulation starts running but after all objects are "ready to go"
        :return:
        """

    def step(self, mcs):
        """
        Called every 'self.frequency' Monte Carlo Steps
        :param mcs: Monte Carlo Step
        :return:
        """

        for cell in self.cell_list:
            print("cell.id=", cell.id)

    def finish(self):
        """
        function called after simulation starts running. Some simulation objects may be inaccessible
        :return:
        """
