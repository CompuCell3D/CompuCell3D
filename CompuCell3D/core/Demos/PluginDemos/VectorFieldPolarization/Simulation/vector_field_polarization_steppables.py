from cc3d.core.PySteppables import *

class VectorFieldSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        fiber_field_cpp = self.field.Fibers

        fiber_field_cpp[:50, :50, 0] = [0, 0, 0]

        fiber_field_cpp[15, 25, 0] = [-15, -25, 0]
        fiber_field_cpp[30, 10, 0] = [-30, -10, 0]
        fiber_field_cpp[10, 32, 0] = [-10, -32, 0]

        fiber_field_cpp[10, 20, 0] = [-10, -20, 0]
        fiber_field_cpp[30, 40, 0] = [-30, -40, 0]
        fiber_field_cpp[20, 30, 0] = [-20, -30, 0]

        int_npy_field_cpp = self.field.intNpyFieldCpp
        int_npy_field_cpp[50:, 50:, 0] = 0


    def step(self, mcs):
        pass