from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCellExtraModules


class HeterotypicBoundarySurfaceSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.htbl_steppable_cpp = None

    def start(self):
        self.htbl_steppable_cpp = CompuCellExtraModules.getHeterotypicBoundaryLength()

    def step(self, mcs):
        self.htbl_steppable_cpp.calculateHeterotypicSurface()

        print(' HTBL between type 1 and 2 is ',
              self.htbl_steppable_cpp.getHeterotypicSurface(1, 2))

        print(' HTBL between type 2 and 1 is ',
              self.htbl_steppable_cpp.getHeterotypicSurface(1, 2))

        print(' HTBL between type 1 and 1 is ',
              self.htbl_steppable_cpp.getHeterotypicSurface(1, 1))

        print(' HTBL between type 0 and 1 is ',
              self.htbl_steppable_cpp.getHeterotypicSurface(0, 1))

        print('THIS ENTRY DOES NOT EXIST. HTBL between type 3 and 20 is ',
              self.htbl_steppable_cpp.getHeterotypicSurface(3, 20))
