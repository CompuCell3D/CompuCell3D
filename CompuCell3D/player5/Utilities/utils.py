def extract_address_int_from_vtk_object(field_extractor, vtkObj):
    '''
    Extracts memory address of vtk object
    :param _vtkObj: vtk object - e.g. vtk array
    :return: int (possible long int) representing the address of the vtk object
    '''
    return field_extractor.unmangleSWIGVktPtrAsLong(vtkObj.__this__)

def qcolor_to_rgba(qcolor):
    """
    Converts qt_color to rgba tuple
    :param qt_color: {QColor}
    :return: {tuple (int, int, int, int)} rgba
    """


    return (qcolor.red(),qcolor.green(),qcolor.blue(),qcolor.alpha())

def to_vtk_rgb(color_obj):
    """

    :param color_obj:{color obj} can be either qcolor or a list/tuple of 3-4 inteegers
    :return: {tuple of 0-1 floats}
    """
    # try qcolor conversion
    try:
        return qcolor_to_rgba(color_obj)[:3]
    except AttributeError:
        pass

    if isinstance(color_obj,list):
        if len(color_obj) < 3:
            raise IndexError ('color_obj list should have at least 3 elements')

        return list(map(lambda x: x/255.0, color_obj))[:3]
    else:
        raise AttributeError('color_obj is of unknown type')