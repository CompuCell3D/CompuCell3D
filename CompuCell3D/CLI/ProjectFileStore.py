projectFilePath = ""
pythonScriptPath = ""
outputDirectoryPath = ""
outputFrequency = ""
parameterScanFile = ""
filePathToResourceTypeMap = {}


class ResourceType:
    """
    This class works as enumeration for the resource types in used in
    CompuCell3D project.
    """
    PYTHON = 1
    XML = 2
    PARAMETER_SCAN = 3

