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

        vector_field_cpp = self.field.Fibers
        vec_field_array = self.field.vector_numpy

        vector_field_cpp[:50, :50, 0] = [0, 0, 0]

        vector_field_cpp[15, 25, 0] = [-15, -25, 0]
        vector_field_cpp[30, 10, 0] = [-30, -10, 0]
        vector_field_cpp[10, 32, 0] = [-10, -32, 0]

        vector_field_cpp[10, 20, 0] = [-10, -20, 0]
        vector_field_cpp[30, 40, 0] = [-30, -40, 0]
        vector_field_cpp[20, 30, 0] = [-20, -30, 0]

        vec_field_array = self.field.vector_numpy
        vec_field_array[15, 25, 0] = [15, 25, 0]
        vec_field_array[30, 10, 0] = [30, 10, 0]
        vec_field_array[10, 32, 0] = [10, 32, 0]

        vec_field_array[10, 20, 0] = [10, 20, 0]
        vec_field_array[30, 40, 0] = [30, 40, 0]
        vec_field_array[20, 30, 0] = [20, 30, 0]


        array = self.field.numpy1


        array[0, 0, 0] = 1000
        array[15, 25, 0] = 12
        array[30, 10, 0] = 30
        array[10, 32, 0] = 3

        array[10, 20, 0] = 120
        array[30, 40, 0] = 300
        array[20, 30, 0] = 311
        array[99, 99, 0] = 1000

        cpp_array = self.raw_field.cpp_numpy
        cpp_array[0, 0, 0] = 1000
        cpp_array[99, 99, 0] = 1000


    def step(self, mcs):
        pg = CompuCellSetup.persistent_globals
        array, aux_field = pg.field_registry.shared_scalar_numpy_fields["numpy1"]
        # aux_field.printAllArrayValues()
        # if mcs > 500:
        #     CompuCellSetup.stopSimulation()
