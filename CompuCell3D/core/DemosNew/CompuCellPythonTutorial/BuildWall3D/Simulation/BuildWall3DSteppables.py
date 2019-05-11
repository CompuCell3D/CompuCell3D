from cc3d.core.PySteppables import *

class BuildWall3DSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.build_wall(self.WALL)

    def step(self, mcs):
        print('MCS=', mcs)
        if mcs == 4:
            self.destroy_wall()
            self.resize_and_shift_lattice(new_size=(80, 80, 80), shift_vec=(10, 10, 10))
        if mcs == 6:
            self.build_wall(self.WALL)
