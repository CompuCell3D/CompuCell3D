import warnings
# from vtkmodules.vtkCommonCorePython import vtkObjectBase
from vtk import vtkObjectBase


def extract_address_int_from_vtk_object(vtkObj) -> int:
    """
    Extracts memory address of vtk object

    :param vtkObj: vtk object - e.g. vtk array
    :return: int (possible long int) representing the address of the vtk object
    """

    addr_portion = vtkObj.__this__.split('_')[1]
    addr_hex_int = int(addr_portion, 16)

    return addr_hex_int


def recover_vtk_object_from_address_int(addr: int):
    """
    Returns the vtk object at an address

    :param addr: address of object, *e.g.*, as returned by :func:`extract_address_int_from_vtk_object`
    :type addr: int
    :return: vtk object at the provided address
    :rtype: vtkObjectBase
    """

    return vtkObjectBase(hex(addr))


def color_to_rgba(color: object) -> tuple:
    """
    Converts color to rgba tuple

    :param color: {Color}
    :return: {tuple (int, int, int, int)} rgba
    """

    return (color.red(), color.green(), color.blue(), color.alpha())


def to_vtk_rgb(color_obj):
    """
    Converts color object (either QColor or a tuple of intergers) into vtk rgb values

    :param color_obj:{color obj} can be either qcolor or a list/tuple of 3-4 integers
    :return: {tuple of 0-1 floats}
    """
    # try qcolor conversion
    try:
        color_rgb = color_to_rgba(color_obj)[:3]
        color_obj = color_rgb
    except AttributeError:
        pass

    if isinstance(color_obj, list) or isinstance(color_obj, tuple):
        if len(color_obj) < 3:
            raise IndexError('color_obj list should have at least 3 elements')

        return list([x / 255.0 for x in color_obj])[:3]
    else:
        raise AttributeError('color_obj is of unknown type')


def cs_string_to_typed_list(cs_str: str, sep=",", type_conv_fcn=float):
    """
    Coinvers comma (or sep) separated string into a list of specific type

    :param cs_str: {str} str to convert
    :param sep: {str} separator  - default is ','
    :param type_conv_fcn: {function} type converting fcn
    :return: {list}
    """
    try:
        list_strings = cs_str.split(sep)
        if all(map(lambda s: s.strip() == '', cs_str.split(sep))):
            # we are getting a list of empty strings we return [] and do not print warning
            return []
        return list([type_conv_fcn(x) for x in list_strings])
    except:
        warnings.warn('Could not convert string {s} to a typed list'.format(s=cs_str))
        return []
