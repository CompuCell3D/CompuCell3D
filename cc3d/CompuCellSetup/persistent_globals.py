from cc3d.core.SteppableRegistry import SteppableRegistry
class PersistentGlobals:
    def __init__(self):
        self.cc3d_xml_2_obj_converter = None
        self.steppable_registry = SteppableRegistry()
        self.sim = None
        self.simthread = None
        self.simulation_initialized = False

    def clean(self):
        """

        :return:
        """
