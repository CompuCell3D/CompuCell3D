from cc3d.cpp import CC3DAuxFields
import numpy as np
import ctypes
from typing import Tuple


cpp_type_to_npy_dtype = {
    "float": np.float32,
    "int": np.int32,
    "double": np.float64,
}

# Mapping NumPy dtype to ctypes types
np_to_ctypes = {
    np.dtype("int8"): ctypes.c_byte,  # signed char
    np.dtype("uint8"): ctypes.c_ubyte,  # unsigned char
    np.dtype("int16"): ctypes.c_short,  # short
    np.dtype("uint16"): ctypes.c_ushort,  # unsigned short
    np.dtype("int32"): ctypes.c_int,  # int
    np.dtype("uint32"): ctypes.c_uint,  # unsigned int
    np.dtype("int64"): ctypes.c_longlong,  # long long
    np.dtype("uint64"): ctypes.c_ulonglong,  # unsigned long long
    np.dtype("float32"): ctypes.c_float,  # float
    np.dtype("float64"): ctypes.c_double,  # double
}


def create_shared_numpy_array_as_cc3d_scalar_field(shape: Tuple, padding=0, dtype=np.float32):
    if dtype in (np.float64,):
        field = CC3DAuxFields.NumpyArrayWrapper3DImplDouble(shape, padding=padding)
        size = field.getSize()
        padding_vec = field.getPaddingVec()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        padded_shape = tuple(np.array(shape) + 2 * np.array(padding_vec))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(padded_shape)
        return array, field
    elif dtype in (np.float32,):
        field = CC3DAuxFields.NumpyArrayWrapper3DImplFloat(shape, padding=padding)
        size = field.getSize()
        padding_vec = field.getPaddingVec()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        padded_shape = tuple(np.array(shape) + 2 * np.array(padding_vec))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(padded_shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def create_field_and_array_from_cc3d_shared_numpy_scalar_field(field):
    element_type = field.getElementType()
    # padding = field.getPadding()

    try:
        dtype = np.dtype(element_type)
    except TypeError:
        raise TypeError(f"shared numpy scalar field element type: '{element_type}' is not supported")

    # if dtype is None:
    #     raise ValueError(f"shared numpy scalar field element type: '{element_type}' is not supported")

    ctypes_type_obj = np_to_ctypes.get(dtype, None)
    if ctypes_type_obj is None:
        raise ValueError(f"Could not associate numpy dtype {dtype} with the corresponding ctypes type object ")
    dim = field.getDim()
    shape = [dim.x, dim.y, dim.z]
    size = field.getSize()

    ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes_type_obj))
    buffer = (ctypes_type_obj * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))
    padded_shape = tuple(np.array(shape))
    array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(padded_shape)
    return array, field


def create_shared_numpy_array_as_cc3d_vector_field(shape: Tuple, dtype=np.float32):
    if dtype in (np.float64,):
        field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplDouble(shape)
        size = field.getSize()

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype in (np.float32,):
        field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplFloat(shape)

        # field = CC3DAuxFields.VectorNumpyArrayWrapper3DImplFloat(shape)
        size = int(field.getSize())
        print("size=", size)
        print("ctypes.c_float=", ctypes.c_float)

        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_float))
        buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    else:
        raise ValueError(f"Unsupported dtype={dtype}")


def create_field_and_array_from_cc3d_vector_field(field):
    element_type = field.getElementType()

    dtype = cpp_type_to_npy_dtype.get(element_type, None)
    if dtype is None:
        raise ValueError(f"vector field element type: '{element_type}' is not supported")

    dim = field.getDim()
    shape = [dim.x, dim.y, dim.z, 3]
    size = field.getSize()

    if dtype == np.float64:
        ptr_as_ctypes = ctypes.cast(int(field.getPtr()), ctypes.POINTER(ctypes.c_double))
        buffer = (ctypes.c_double * size).from_address(ctypes.addressof(ptr_as_ctypes.contents))

        array = np.frombuffer(buffer, dtype=dtype, count=size).reshape(shape)
        return array, field
    elif dtype == np.float32:
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
