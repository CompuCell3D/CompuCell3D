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

        numpy_field_manager = self.field.numpy_field_manager

        cell_type_field = self.field.cell_type_field
        cell_type_field[10,10,0] = 20
        cell_type_field[10, 20, 0] = 40
        cell_type_field[50, 50, 0] = 250
        # initializing FGF diffusion field
        fgf = self.field.FGF
        fgf[50, 50, 0] = 2000


        fibers_cpp = self.field.Fibers

        fibers_cpp[:50, :50, 0] = [0, 0, 0]

        fibers_cpp[15, 25, 0] = [-15, -25, 0]
        fibers_cpp[30, 10, 0] = [-30, -10, 0]
        fibers_cpp[10, 32, 0] = [-10, -32, 0]

        fibers_cpp[10, 20, 0] = [-10, -20, 0]
        fibers_cpp[30, 40, 0] = [-30, -40, 0]
        fibers_cpp[20, 30, 0] = [-20, -30, 0]

        vec_field_array = self.field.vector_numpy
        vec_field_array[15, 25, 0] = [15, 25, 0]
        vec_field_array[30, 10, 0] = [30, 10, 0]
        vec_field_array[10, 32, 0] = [10, 32, 0]

        vec_field_array[10, 20, 0] = [10, 20, 0]
        vec_field_array[30, 40, 0] = [30, 40, 0]
        vec_field_array[20, 30, 0] = [20, 30, 0]

        # self.field accesses array in coordinates that exclude padded region
        array = self.field.numpy1


        array[0, 0, 0] = 1000
        array[15, 25, 0] = 12
        array[30, 10, 0] = 30
        array[10, 32, 0] = 3

        array[10, 20, 0] = 120
        array[30, 40, 0] = 300
        array[20, 30, 0] = 311
        array[99, 99, 0] = 1000

        # accessing scalar field created in C++
        # self.raw_field access "raw array" i.e. in coordinates where you can modify padded region
        cpp_array = self.raw_field.cpp_numpy
        cpp_array[0, 0, 0] = 1000
        cpp_array[99, 99, 0] = 1000

        cpp_array_user = self.field.cpp_numpy
        cpp_array_user[0, 0, 0] = 1000
        cpp_array_user[99, 99, 0] = 1000



        # accessing scalar field created in C++ using FieldManager
        np_fm_array = self.field.numpy_field_manager

        np_fm_array[0, 0, 0] = 1020
        np_fm_array[15, 25, 0] = 1200
        np_fm_array[30, 10, 0] = 3000
        np_fm_array[10, 32, 0] = 30

        np_fm_array[10, 20, 0] = 1205
        np_fm_array[30, 40, 0] = 3005
        np_fm_array[20, 30, 0] = 31
        np_fm_array[99, 99, 0] = 100

        # self.raw_field access "raw array" i.e. in coordinates wher you can modify padded region
        np_fm_raw = self.raw_field.numpy_field_manager
        np_fm_raw[0, 0, 0] = 1010
        np_fm_raw[99, 99, 0] = 1010


        # accessing vector field created in C++ using FieldManager
        fibers_fm = self.field.fibers_field_manager
        fibers_fm[0, 0, 0,...] = [1020, 1020, 0]
        fibers_fm[15, 25, 0,...] = [120, 120, 0]
        fibers_fm[30, 10, 0,...] = [3000, 400,0]
        fibers_fm[10, 32, 0,...] = [30,-30,0]


        fibers_fm

    def step(self, mcs):
        pg = CompuCellSetup.persistent_globals
        array, aux_field = pg.field_registry.shared_scalar_numpy_fields["numpy1"]
        # aux_field.printAllArrayValues()
        # if mcs > 500:
        #     CompuCellSetup.stopSimulation()
