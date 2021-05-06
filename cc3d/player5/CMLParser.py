# todo: abstract current connectivity to Twedit

import os.path
import argparse


class CMLParser(object):
    def __init__(self):
        self.__screenshotDescriptionFileName = ''
        self.customScreenshotDirectoryName = ''
        self.__fileName = ''
        self.__cml_args = None
        self.outputFrequency = 1
        self.outputFileCoreName = ''

    @property
    def fileName(self):
        return self.__fileName

    @property
    def cml_args(self):
        return self.__cml_args

    @property
    def screenshotDescriptionFileName(self):
        return self.__screenshotDescriptionFileName

    def getSimulationFileName(self):
        return self.__fileName

    def parse_cml(self, arg_list=None):
        """
        Parses command line
        :return:
        """

        cml_parser = argparse.ArgumentParser(description='CompuCell3D Player 5')
        cml_parser.add_argument('-i', '--input', required=False, action='store',
                                help='path to the CC3D project file (*.cc3d)')

        cml_parser.add_argument('-c', '--output-file-core-name', required=False, action='store',
                                help='core name for vtk files.')

        cml_parser.add_argument('--noOutput', required=False, action='store_true', default=False,
                                help='flag suppressing output of simulation snapshots')
        cml_parser.add_argument('-f', '--outputFrequency', required=False, action='store',
                                default=0, type=int,
                                help='simulation snapshot output frequency')

        cml_parser.add_argument('--output-frequency', required=False, action='store', type=int, default=0,
                                help='simulation snapshot output frequency')

        cml_parser.add_argument('--screenshot-output-frequency', required=False, action='store', type=int, default=-1,
                                help='screenshot output frequency')

        cml_parser.add_argument('-s', '--screenshotDescription', required=False, action='store',
                                help='screenshot description file name (deprecated)')

        cml_parser.add_argument('--currentDir', required=False, action='store',
                                help='current working directory')

        cml_parser.add_argument('--numSteps', required=False, action='store', default=False, type=int,
                                help='overwrites number of Monte Carlo Steps that simulation will run for')

        cml_parser.add_argument('--testOutputDir', required=False, action='store',
                                help='test output directory (used during unit testing only)')

        cml_parser.add_argument('-o', '--screenshotOutputDir', required=False, action='store',
                                help='directory where screenshots should be written to')

        cml_parser.add_argument('--output-dir', required=False, action='store',
                                help='directory where screenshots should be written to')

        cml_parser.add_argument('-p', '--playerSettings', required=False, action='store',
                                help='file with player settings (deprecated)')

        cml_parser.add_argument('-w', '--windowSize', required=False, action='store',
                                help='specifies window size Format is  WIDTHxHEIGHT e.g. -w 500x300 (deprecated)')

        cml_parser.add_argument('--port', required=False, action='store', type=int,
                                help='specifies listening port for communication with Twedit')

        cml_parser.add_argument('--tweditPID', required=False, action='store', type=int,
                                help='process id for Twedit')

        cml_parser.add_argument('--prefs', required=False, action='store',
                                help='specifies path tot he Qt settings file for Player (debug mode only)')

        cml_parser.add_argument('--exitWhenDone', required=False, action='store_true', default=False,
                                help='exits Player at the end of the simulation')

        cml_parser.add_argument('--exit-when-done', required=False, action='store_true', default=False,
                                help='exits Player at the end of the simulation')

        cml_parser.add_argument('--guiScan', required=False, action='store_true', default=False,
                                help='enables running parameter scan in the Player')

        cml_parser.add_argument('--parameter-scan-iteration', required=False, type=str, default='',
                                help='optional argument that specifies parameter scan iteration - '
                                     'used to enable steppables and player gui (to display in the title window)'
                                     'to access current param scan iteration number')

        cml_parser.add_argument('--maxNumberOfConsecutiveRuns', required=False, action='store', default=0, type=int,
                                help='maximum number of consecutive runs in the Player before Player restarts')

        cml_parser.add_argument('--pushAddress', required=False, action='store',
                                help='address used to push data from the worker (optimization runs only)')

        cml_parser.add_argument('--returnValueTag', required=False, action='store',
                                help='return value tag (optimization runs only))')

        if arg_list is None:
            arg_list = []

        self.__cml_args = cml_parser.parse_args(arg_list)

        # handling multiple versions of long options
        if self.__cml_args.output_frequency is not None:
            self.__cml_args.outputFrequency = self.__cml_args.output_frequency

        if self.__cml_args.output_dir is not None:
            self.__cml_args.screenshotOutputDir = self.__cml_args.output_dir

        if self.__cml_args.exit_when_done:
            self.__cml_args.exitWhenDone = self.__cml_args.exit_when_done

        # filling out legacy variables

        self.__fileName = self.__cml_args.input
        if self.__fileName:
            if self.__cml_args.output_file_core_name is not None:
                self.outputFileCoreName = self.__cml_args.output_file_core_name
            else:
                self.outputFileCoreName = os.path.basename(self.__fileName.replace('.', '_'))

        self.__screenshotDescriptionFileName = self.__cml_args.screenshotDescription
        self.customScreenshotDirectoryName = self.__cml_args.screenshotOutputDir

        self.outputFrequency = self.__cml_args.outputFrequency

        return False
