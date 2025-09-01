from cc3d.core.PySteppables import *


class SharedNUmpyFieldsSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        # adding padding of 1 around the field to allow development of PDE solvers
        self.pad = 1
        self.create_shared_scalar_numpy_field("numpy1", padding=self.pad)
        self.create_shared_vector_numpy_field("vector_numpy")

        self.create_shared_scalar_numpy_field("int16FieldPythonNPY", precision_type="int16")
        self.create_shared_scalar_numpy_field("float32FieldPythonNPY", precision_type="float32")

    def start(self):
        int16FieldPythonNPY = self.field.int16FieldPythonNPY
        int16FieldPythonNPY[20:30, 20:30, 0] = 20
        int16FieldPythonNPY[30:40, 30:40, 0] = 30

        float32FieldPythonNPY = self.field.float32FieldPythonNPY

        float32FieldPythonNPY[70:80, 70:80, 0] = 20.2
        float32FieldPythonNPY[80:90, 80:90, 0] = 30.2

        self.copy_cell_attribute_field_values_to("cell_type_field", "type")
        self.copy_cell_attribute_field_values_to("cell_volume_field", "id")


        numpy_field_manager = self.field.numpy_field_manager

        cell_type_field = self.field.cell_type_field
        cell_volume_field = self.field.cell_volume_field
        # cell_volume_field[45:50, 45:50, 0] = 25

        # cell_type_field[5:10,5:10,0] = 20
        # cell_type_field[5:10, 15:20, 0] = 40
        # cell_type_field[45:50, 45:50, 0] = 250
        # print("cell_type_field[50,50,0]=", cell_type_field[50,50,0])
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
        fibers_fm[0, 0, 0, ...] = [1020, 1020, 0]
        fibers_fm[15, 25, 0, ...] = [120, 120, 0]
        fibers_fm[30, 10, 0, ...] = [3000, 400, 0]
        fibers_fm[10, 32, 0, ...] = [30, -30, 0]


    def step(self, mcs):
        # demonstrating how we can quickly copy legacy concentration fields from C++ CC3D to shared numpy array
        # note, the field destination_field_name must be of type float32 :
        # <Field Name="concentration_field_copy" Type="scalar" Precision="float32"/>
        self.copy_legacy_concentration_field(source_field_name="FGF", destination_field_name="concentration_field_copy")

        pass

