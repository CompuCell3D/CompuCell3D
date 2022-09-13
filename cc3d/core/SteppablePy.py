class SteppablePy:
    """
    Steppable model specification interface
    All CC3D model specification using a steppable should be done by overriding the methods `start`, `step`, `on_stop`
    and `finish`
    """

    def __init__(self):
        #: flag to step before engine when equal to 1
        self.runBeforeMCS = 0

    def core_init(self):
        """

        :return:
        """

    def start(self):
        """
        any code in the start function runs before MCS=0

        :return: None
        """

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS

        :param mcs: current Monte Carlo step
        :return: None
        """

    def finish(self):
        """
        Finish Function is called after the last MCS

        :return: None
        """
    def on_stop(self):
        """
        Called when simulation is stopped by user

        :return: None
        """

    def cleanup(self):
        """

        :return:
        """
