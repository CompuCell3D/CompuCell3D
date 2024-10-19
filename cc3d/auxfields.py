from cc3d.cpp import CompuCell
from cc3d.cpp import CC3DAuxFields
import numpy as np
import ctypes


def main():
    print("aux fields")
    dim_x = 4
    dim_y = 6

    aux_field = CC3DAuxFields.NumpyArrayWrapper(dim_x, dim_y)
    size = aux_field.getSize()
    print("size=", aux_field.getSize())

    aux_field.setDimensions((10,20,30,40))
    # print("dimensions=", aux_field.getDimensions())

    return




    ptr = aux_field.getPtr()

    print("ptr=", ptr)
    ptr_as_ctypes = ctypes.cast(int(ptr), ctypes.POINTER(ctypes.c_double))

    # ptr_as_ctypes = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    print("ptr_as_ctypes=", ptr_as_ctypes)
    buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))
    # array = np.frombuffer(ptr, dtype=np.float64, count=200)
    array = np.frombuffer(buffer, dtype=np.float64, count=size).reshape((dim_x, dim_y))

    # print(array[:10])
    array[1,2] = 12
    array[3, 0] = 30
    array[0, 3] = 3

    aux_field.printArray()


if __name__ == '__main__':
    main()