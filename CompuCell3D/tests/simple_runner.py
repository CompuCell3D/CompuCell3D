import argparse
import sys
import CompuCellSetup
import CC3DSimulationDataHandler as CC3DSimulationDataHandler
from os import environ
import os

def process_cml():
    cml_parser = argparse.ArgumentParser(description='param_scan_run - Parameter Scan Run Script')
    cml_parser.add_argument('-i', '--input', required=True, action='store',
                            help='path .cc3d script (*.cc3d)')
    cml_parser.add_argument('-o', '--output', required=True, action='store',
                            help='path to the output folder to store parameter scan results')
    cml_parser.add_argument('-f', '--output-frequency', required=False, action='store', default=1, type=int,
                            help='simulation snapshot output frequency')

    args = cml_parser.parse_args()

    return args

def readCC3DFile(fileName):

    cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.readCC3DFileFormat(fileName)
    print cc3dSimulationDataHandler.cc3dSimulationData

    return cc3dSimulationDataHandler

def prepareSingleRun(_cc3dSimulationDataHandler):

    CompuCellSetup.simulationPaths.setSimulationBasePath(_cc3dSimulationDataHandler.cc3dSimulationData.basePath)

    if _cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":

        CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(
            _cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)

        if _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(
                _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)


    elif _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":

        CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(
            _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)

def set_environment_vars():

    def setVTKPaths():
        platform = sys.platform
        if platform == 'win32':
            sys.path.insert(0, environ["PYTHON_DEPS_PATH"])


    setVTKPaths()

    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)

    swig_lib_install_path = os.environ["SWIG_LIB_INSTALL_DIR"]
    appended = sys.path.count(swig_lib_install_path)
    if not appended:
        sys.path.append(swig_lib_install_path)

    import vtk
    # even if vtk is not use it needs to be imported before other swig modules (PlayerPython) can be used



if __name__ == '__main__':

    # sets necessary environment variables
    set_environment_vars()

    args = process_cml()
    cc3d_project_fname = args.input
    output_dir = args.output
    output_frequency = args.output_frequency

    # necessary global setting
    CompuCellSetup.simulationFileName = cc3d_project_fname

    # read content of .cc3d file
    cc3dSimulationDataHandler = readCC3DFile(cc3d_project_fname)

    # set necessary CompuCellSetup global
    CompuCellSetup.cc3dSimulationDataHandler = cc3dSimulationDataHandler

    # copies simulation scripts paths to another object - CompuCellSetup.simulationPaths (do not ask why)
    prepareSingleRun(cc3dSimulationDataHandler)

    # setting simuation output folder
    CompuCellSetup.simulationPaths.simulationResultStorageDirectory = output_dir

    # setting player type - indicaets that we will use mainLookCML from CompuCellSetup
    CompuCellSetup.playerType = "CML"

    # setting output frequency
    CompuCellSetup.cmlParser.outputFrequency = output_frequency

    # executes python script from the simulation
    execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
