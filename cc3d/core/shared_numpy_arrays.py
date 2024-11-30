from cc3d.cpp import CC3DAuxFields
import numpy as np
import ctypes
from typing import Tuple


cpp_type_to_npy_dtype = {
    "float": np.float32,
    "int": np.int32,
    "double": np.float64,
}

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


def create_shared_numpy_array_as_cc3d_scalar_field(shape: Tuple, dtype=np.float32):
    if dtype in (np.float64,):
        field = CC3DAuxFields.NumpyArrayWrapper3DImplDouble(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype in (np.float32,):
        field = CC3DAuxFields.NumpyArrayWrapper3DImplFloat(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def create_shared_numpy_array_as_cc3d_vector_field(shape: Tuple, dtype=np.float32):
    if dtype in (np.float64,):
        field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplDouble(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype in (np.float32,):
        field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplFloat(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        # array = np.frombuffer(ptr, dtype=np.float64, count=200)
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")

def create_field_and_array_from_cc3d_vector_field(field: CC3DAuxFields.VectorNumpyArrayWrapper3DImplDouble):
    element_type = field.getElementType()

    dtype = cpp_type_to_npy_dtype.get(element_type, None)
    if dtype is None:
        raise ValueError(f"vector field element type: '{element_type}' is not supported")

    dim = field.getDim()
    shape = [dim.x, dim.y, dim.z, 3]
    size = field.getSize()

    if dtype in (np.float64,):
        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype in (np.float32,):

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))
        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def register_shared_numpy_array(field_name: str, simulator, dtype=np.float32):
    potts = simulator.getPotts()
    dim = potts.getCellFieldG().getDim()
    array, field = create_shared_numpy_array_as_cc3d_scalar_field(shape=(dim.x, dim.y, dim.z), dtype=dtype)
    simulator.registerConcentrationField(field_name, field)
    return array, field
