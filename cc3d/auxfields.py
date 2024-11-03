from cc3d.cpp import CompuCell
from cc3d.cpp import CC3DAuxFields
import numpy as np
import ctypes
from typing import Tuple


def get_shared_numpy_array(shape: Tuple, dtype=np.float64):
    if dtype in (np.float64,):
        field = CC3DAuxFields.NumpyArrayWrapperImplDouble(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype in (np.float32,):
        field = CC3DAuxFields.NumpyArrayWrapperImplFloat(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def main():

    print("aux fields")

    dim_x = 4
    dim_y = 6
    dim_z = 5
    shape = (dim_x, dim_y, dim_z)
    # shape = (dim_x, dim_y)
    # shape = (30, )
    wrapper = CC3DAuxFields.NumpyArrayWrapper3DImplFloat(shape)

    array, aux_field_double = get_shared_numpy_array(shape=shape, dtype=np.float64)
    array[1, 2, 0] = 12
    array[3, 0, 0] = 30
    array[0, 3, 0] = 3

    array[1, 2, 3] = 120
    array[3, 0, 2] = 300
    array[2, 3, 4] = 311
    aux_field_double.printAllArrayValues()

    # aux_field_double = CC3DAuxFields.NumpyArrayWrapperImplDouble(shape)
    # aux_field_double.printAllArrayValues()

    aux_field = CC3DAuxFields.NumpyArrayWrapper(shape)

    # aux_field.printAllArrayValues()

    # aux_field.iterateOverAxes((2,3,2))

    size = aux_field.getSize()
    print("size=", aux_field.getSize())

    # aux_field.setDimensions((10, 20, 30, 40))
    # print("dimensions=", aux_field.getDimensions())

    ptr = aux_field.getPtr()

    print("ptr=", ptr)
    ptr_as_ctypes = ctypes.cast(int(ptr), ctypes.POINTER(ctypes.c_double))

    # ptr_as_ctypes = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    print("ptr_as_ctypes=", ptr_as_ctypes)
    buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

    # array = np.frombuffer(ptr, dtype=np.float64, count=200)
    array = np.frombuffer(buffer, dtype=np.float64, count=size).reshape(shape)

    # print(array[:10])

    # array[10] = 12
    # array[21] = 30
    # array[28] = 3

    # array[1, 2] = 12
    # array[3, 0] = 30
    # array[0, 3] = 3

    array[1, 2, 0] = 12
    array[3, 0, 0] = 30
    array[0, 3, 0] = 3

    array[1, 2, 3] = 120
    array[3, 0, 2] = 300
    array[2, 3, 4] = 311

    aux_field.printAllArrayValues()
    # aux_field.printArray()


def main_vector():
    print("aux vector fields")

    dim_x = 4
    dim_y = 6
    dim_z = 5
    shape = (dim_x, dim_y, dim_z)
    # shape = (dim_x, dim_y)
    # shape = (30, )

    pt = CompuCell.Point3D(2,3,4)

    dtype = np.float32
    # scalar field - just to check if swig wrapper works
    wrapper = CC3DAuxFields.NumpyArrayWrapper3DImplFloat(shape)



    vec_shape = (dim_x, dim_y, dim_z, 3)
    cc3d_cpp_vec_field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplFloat(vec_shape)
    # cc3d_cpp_vec_field.set()

    size = cc3d_cpp_vec_field.getSize()

    ptr_as_ctypes = ctypes.cast(int(cc3d_cpp_vec_field.getPtr()), ctypes.POINTER(ctypes.c_float))
    buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

    # array = np.frombuffer(ptr, dtype=np.float64, count=200)
    vec_array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(vec_shape)


    vec_array[1, 2, 3, ...] = [20, 30, 40]

    print("vector 1,2, 2=", vec_array[1, 2, 3])

    cc3d_cpp_vec_field.set(CompuCell.Point3D(1, 0, 4), CC3DAuxFields.cc3dauxfield_coordinates3d_float(111, 12, 13))

    print("check=", vec_array[1, 0, 4])

    coord_3d = cc3d_cpp_vec_field.get(CompuCell.Point3D(1, 2, 3))
    print("coord_3d=", coord_3d)
    print("coord_3d.x=", coord_3d.X())
    print("coord_3d.y=", coord_3d.y)
    print("coord_3d.z=", coord_3d.z)

    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                print(f"{x}, {y}, {z}=", vec_array[x,y,z])



if __name__ == "__main__":
    main_vector()
    # main()
