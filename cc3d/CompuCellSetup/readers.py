from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationDataHandler

def readCC3DFile(fileName):
    """
    reads .cc3d file
    :param fileName:{str}
    :return: {object: cc3dSimulationDataHandler}
    """

    cc3dSimulationDataHandler = CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.read_cc3_d_file_format(fileName)
    print(cc3dSimulationDataHandler.cc3dSimulationData)

    return cc3dSimulationDataHandler
