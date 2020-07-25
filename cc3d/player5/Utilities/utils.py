import warnings


def extract_address_int_from_vtk_object(vtkObj: str) -> int:
    '''
    Extracts memory address of vtk object
    :param _vtkObj: vtk object - e.g. vtk array
    :return: int (possible long int) representing the address of the vtk object
    '''

    addr_portion = vtkObj.__this__.split('_')[1]
    addr_hex_int = int(addr_portion, 16)

    return addr_hex_int


def qcolor_to_rgba(qcolor: object) -> tuple:
    """
    Converts qt_color to rgba tuple
    :param qt_color: {QColor}
    :return: {tuple (int, int, int, int)} rgba
    """

    return (qcolor.red(), qcolor.green(), qcolor.blue(), qcolor.alpha())


def to_vtk_rgb(color_obj):
    """
    Converts color object (either QColor or a tuple of intergers) into vtk rgb values
    :param color_obj:{color obj} can be either qcolor or a list/tuple of 3-4 integers
    :return: {tuple of 0-1 floats}
    """
    # try qcolor conversion
    try:
        qcolor_rgb = qcolor_to_rgba(color_obj)[:3]
        color_obj = qcolor_rgb
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
