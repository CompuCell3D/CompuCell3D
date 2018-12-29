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

    def init(self, _simulator):
        """

        :param _simulator:
        :return:
        """