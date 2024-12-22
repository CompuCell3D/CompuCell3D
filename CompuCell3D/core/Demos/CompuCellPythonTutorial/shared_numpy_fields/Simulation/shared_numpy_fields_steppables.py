from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup


class SharedNUmpyFieldsSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        # adding padding of 1 around the field to allow development of PDE solvers
        self.pad = 1

        self.create_shared_scalar_numpy_field("numpy1", padding=self.pad)
        self.create_shared_vector_numpy_field("vector_numpy")

    def start(self):
        #
        # vector_field_cpp = self.field.Fibers
        # vec_field_array = self.field.vector_numpy
        #
        # vector_field_cpp[:50, :50, 0] = [0, 0, 0]
        #
        # vector_field_cpp[15, 25, 0] = [-15, -25, 0]
        # vector_field_cpp[30, 10, 0] = [-30, -10, 0]
        # vector_field_cpp[10, 32, 0] = [-10, -32, 0]
        #
        # vector_field_cpp[10, 20, 0] = [-10, -20, 0]
        # vector_field_cpp[30, 40, 0] = [-30, -40, 0]
        # vector_field_cpp[20, 30, 0] = [-20, -30, 0]
        #
        # vec_field_array = self.field.vector_numpy
        # vec_field_array[15, 25, 0] = [15, 25, 0]
        # vec_field_array[30, 10, 0] = [30, 10, 0]
        # vec_field_array[10, 32, 0] = [10, 32, 0]
        #
        # vec_field_array[10, 20, 0] = [10, 20, 0]
        # vec_field_array[30, 40, 0] = [30, 40, 0]
        # vec_field_array[20, 30, 0] = [20, 30, 0]

        pad = self.pad

        array = self.field.numpy1
        # if pad > 0:
        #     array_view = array[pad:-pad, pad:-pad, pad:-pad]
        # else:
        #     array_view = array

        array_view = array
        array_view[0, 0, 0] = 1000
        array_view[15, 25, 0] = 12
        array_view[30, 10, 0] = 30
        array_view[10, 32, 0] = 3

        array_view[10, 20, 0] = 120
        array_view[30, 40, 0] = 300
        array_view[20, 30, 0] = 311
        array_view[99, 99, 0] = 1000

        cpp_array = self.field.cpp_numpy
        # cpp_array[0, 0, 0] = 1000
        # cpp_array[99, 99, 0] = 1000

        pad = 1
        # cpp_array_view = cpp_array[pad:-pad, pad:-pad, pad:-pad]
        cpp_array_view = cpp_array
        cpp_array_view[0, 0, 0] = 1000
        cpp_array_view[99, 99, 0] = 1000

    def step(self, mcs):
        pg = CompuCellSetup.persistent_globals
        array, aux_field = pg.field_registry.shared_scalar_numpy_fields["numpy1"]
        # aux_field.printAllArrayValues()
        # if mcs > 500:
        #     CompuCellSetup.stopSimulation()
