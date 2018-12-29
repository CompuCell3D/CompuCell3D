import argparse
import cc3d.CompuCellSetup as CompuCellSetup
from cc3d.CompuCellSetup.readers import readCC3DFile


def process_cml():
    """

    :return:
    """


    cml_parser = argparse.ArgumentParser(description='CompuCell3D Player 5')
    cml_parser.add_argument('-i', '--input', required=False, action='store',
                            help='path to the CC3D project file (*.cc3d)')
    # cml_parser.add_argument('--noOutput', required=False, action='store_true', default=False,
    #                         help='flag suppressing output of simulation snapshots')
    # cml_parser.add_argument('-f', '--outputFrequency', required=False, action='store', default=1, type=int,
    #                         help='simulation snapshot output frequency')
    #
    # cml_parser.add_argument('-s', '--screenshotDescription', required=False, action='store',
    #                         help='screenshot description file name (deprecated)')
    #
    # cml_parser.add_argument('--currentDir', required=False, action='store',
    #                         help='current working directory')
    #
    # cml_parser.add_argument('--numSteps', required=False, action='store', default=False, type=int,
    #                         help='overwrites number of Monte Carlo Steps that simulation will run for')
    #
    # cml_parser.add_argument('--testOutputDir', required=False, action='store',
    #                         help='test output directory (used during unit testing only)')
    #
    # cml_parser.add_argument('-o', '--screenshotOutputDir', required=False, action='store',
    #                         help='directory where screenshots should be written to')
    #
    # cml_parser.add_argument('-p', '--playerSettings', required=False, action='store',
    #                         help='file with player settings (deprecated)')
    #
    # cml_parser.add_argument('-w', '--windowSize', required=False, action='store',
    #                         help='specifies window size Format is  WIDTHxHEIGHT e.g. -w 500x300 (deprecated)')
    #
    # cml_parser.add_argument('--port', required=False, action='store', type=int,
    #                         help='specifies listening port for communication with Twedit')
    #
    # cml_parser.add_argument('--tweditPID', required=False, action='store', type=int,
    #                         help='process id for Twedit')
    #
    # cml_parser.add_argument('--prefs', required=False, action='store',
    #                         help='specifies path tot he Qt settings file for Player (debug mode only)')
    #
    # cml_parser.add_argument('--exitWhenDone', required=False, action='store_true', default=False,
    #                         help='exits Player at the end of the simulation')
    #
    # cml_parser.add_argument('--guiScan', required=False, action='store_true', default=False,
    #                         help='enables running parameter scan in the Player')
    #
    # cml_parser.add_argument('--maxNumberOfConsecutiveRuns', required=False, action='store', default=0, type=int,
    #                         help='maximum number of consecutive runs in the Player before Player restarts')
    #
    # cml_parser.add_argument('--pushAddress', required=False, action='store',
    #                         help='address used to push data from the worker (optimization runs only)')
    #
    # cml_parser.add_argument('--returnValueTag', required=False, action='store',
    #                         help='return value tag (optimization runs only))')


    return cml_parser.parse_args()


if __name__ =='__main__':
    args = process_cml()
    sim_fname = args.input
    cc3dSimulationDataHandler = readCC3DFile(fileName=sim_fname)

    CompuCellSetup.cc3dSimulationDataHandler = cc3dSimulationDataHandler

    # execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
    with open(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript) as sim_fh:
        exec(sim_fh.read())
    # execfile()
    # print



