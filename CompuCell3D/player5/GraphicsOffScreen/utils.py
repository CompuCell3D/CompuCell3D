def extractAddressIntFromVtkObject(field_extractor, vtkObj):
    '''
    Extracts memory address of vtk object
    :param _vtkObj: vtk object - e.g. vtk array
    :return: int (possible long int) representing the address of the vtk object
    '''
    return field_extractor.unmangleSWIGVktPtrAsLong(vtkObj.__this__)