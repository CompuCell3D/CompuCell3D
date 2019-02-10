from cc3d.core.SteppableRegistry import SteppableRegistry
class PersistentGlobals:
    def __init__(self):
        self.cc3d_xml_2_obj_converter = None
        self.steppable_registry = SteppableRegistry()

        # c++ object reference Simulator.cpp
        self.simulator = None

        #  Simulation Thread - either from the player or from CML
        self.simthread = None

        # hook to player - initialized in the player in the prepareForNewSimulation method
        # this object is not None only when GUI run is requested
        self.view_manager = None

        self.simulation_initialized = False
        self.simulation_file_name = None
        self.user_stop_simulation_flag = False


    def clean(self):
        """

        :return:
        """
