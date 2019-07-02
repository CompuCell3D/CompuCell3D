from cc3d.core.PySteppables import *


class DynamicNumberOfProcessorsSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs == 10:
            self.resize_and_shift_lattice(new_size=(400, 400, 1), shift_vec=(100, 100, 0))

        if mcs == 100:
            self.change_number_of_work_nodes(8)
