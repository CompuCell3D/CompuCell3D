import argparse
import ProjectFileStore

class CommandLineArgumentParser:

    def __init__(self):
        self.__argumentParser =  None
        self.__isInitialized = False


    def initialize(self):
        """
        This method initialized creates the argument parser and and initializes with the arguments.

        :return: None
        """
        argumentParser = argparse.ArgumentParser(description='*** CompuCell3D Command-Line Interface ***')

        argumentParser.add_argument('-i', '--input', required=True, action='store',
                                    help='path to the CC3D project file (*.cc3d)')

        argumentParser.add_argument('-o', '--outputDir', required=True, action='store',
                                    help='output directory path to store results')

        argumentParser.add_argument('-f', '--outputFrequency', required=False, action='store', default=1, type=int,
                                    help='simulation snapshot output frequency')

        argumentParser.add_argument('-p', '--parameterFile', required=False, action='store',
                                    help='parameter specification file for parameter scan')

        self.__argumentParser = argumentParser
        self.__isInitialized = True

    def parseArguments(self):
        """
        This method uses the parser created in the initialize method and parse the arguments
        and store the arguments in specified variables

        :return: None
        """
        try:
            if not self.__isInitialized:
                self.initialize()

            arguments, unknown = self.__argumentParser.parse_known_args()

            ProjectFileStore.projectFilePath = arguments.input
            ProjectFileStore.outputDirectoryPath = arguments.outputDir
            ProjectFileStore.outputFrequency = arguments.outputFrequency
            ProjectFileStore.parameterScanFile = arguments.parameterFile

        except SystemExit:
            print '\nError: Invalid command-line arguments. Please refer usage for available options.'
            exit(1)
