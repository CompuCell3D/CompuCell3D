import sys
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

    def processCommandLineOptions(self):

        cml_parser = argparse.ArgumentParser(description='CompuCell3D Player 5')
        cml_parser.add_argument('-i', '--input', required=False, action='store',
                                help='path to the CC3D project file (*.cc3d)')
        cml_parser.add_argument('--noOutput', required=False, action='store_true', default=False,
                                help='flag suppressing output of simulation snapshots')
        cml_parser.add_argument('-f', '--outputFrequency', required=False, action='store', default=1, type=int,
                                help='simulation snapshot output frequency')

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

        cml_parser.add_argument('--guiScan', required=False, action='store_true', default=False,
                                help='enables running parameter scan in the Player')

        cml_parser.add_argument('--maxNumberOfConsecutiveRuns', required=False, action='store', default=0, type=int,
                                help='maximum number of consecutive runs in the Player before Player restarts')

        cml_parser.add_argument('--pushAddress', required=False, action='store',
                                help='address used to push data from the worker (optimization runs only)')

        cml_parser.add_argument('--returnValueTag', required=False, action='store',
                                help='return value tag (optimization runs only))')


        self.__cml_args = cml_parser.parse_args()

        #filling out legacy variables
        self.__fileName = self.__cml_args.input
        if self.__fileName:
            self.outputFileCoreName = os.path.basename(self.__fileName.replace('.', '_'))

        self.__screenshotDescriptionFileName = self.__cml_args.screenshotDescription
        self.customScreenshotDirectoryName = self.__cml_args.screenshotOutputDir
        self.outputFrequency = self.__cml_args.outputFrequency

        #
        #
        #
        #
        # import getopt
        # self.__screenshotDescriptionFileName = ""
        # self.customScreenshotDirectoryName = ""
        # self.outputFrequency = 1
        # self.outputFileCoreName = "Step"
        # self.exitWhenDone = False
        # self.maxNumberOfRuns = -1
        # self.push_address = None
        # self.return_value_tag = 'generic_label'
        # startSimulation = False
        #
        # opts = None
        # args = None
        # try:
        #     opts, args = getopt.getopt(sys.argv[1:], "i:s:o:f:p:l:c:h",
        #                                ["help", "noOutput", "exitWhenDone", "currentDir=", "outputFrequency=",
        #                                 "pushAddress=", "returnValueTag", "maxNumberOfRuns="])
        #     print "opts=", opts
        #     print "args=", args
        # except getopt.GetoptError, err:
        #     # print help information and exit:
        #     print str(err)  # will print something like "option -a not recognized"
        #     # self.usage()
        #     sys.exit(1)
        # output = None
        # verbose = False
        # currentDir = ""
        # for o, a in opts:
        #     print "o=", o
        #     print "a=", a
        #     if o in ("-i"):
        #         self.__fileName = a
        #         startSimulation = True
        #
        #     elif o in ("-h", "--help"):
        #         self.usage()
        #         return True  # help only
        #     elif o in ("-s"):
        #         self.__screenshotDescriptionFileName = a
        #     elif o in ("-c"):
        #         self.outputFileCoreName = a
        #     elif o in ("-o"):
        #         self.customScreenshotDirectoryName = a
        #         self.__noOutput = False
        #     elif o in ("--noOutput"):
        #         self.__noOutput = True
        #         self.outputFrequency = 0
        #     elif o in ("-f", "--outputFrequency"):
        #         self.outputFrequency = int(a)
        #
        #     elif o in ("-p", "--pushAddress"):
        #         self.push_address = str(a)
        #         # print 'GOT pushAddress = ',self.push_address
        #
        #     elif o in ("-l", "--returnValueTag"):
        #         self.return_value_tag = str(a)
        #         print 'GOT return_value_tag = ', self.return_value_tag
        #
        #     elif o in ("--currentDir"):
        #         currentDir = a
        #         print "currentDirectory=", currentDir
        #     elif o in ("--exitWhenDone"):
        #         self.exitWhenDone = True
        #     elif o in ("--maxNumberOfRuns"):
        #         self.maxNumberOfRuns = int(a)
        #
        #
        #         # elif o in ("--exitWhenDone"):
        #         # self.closePlayerAfterSimulationDone=True
        #
        #     else:
        #         assert False, "unhandled option"

        return False

    # def usage(self):
    #     print "USAGE: ./runScript.sh -i <simulation file>  -c <outputFileCoreName> "
    #     print "-o <customVtkDirectoryName>  -f,--frequency=<frequency> "
    #     print "--noOutput will ensure that no output is stored"
    #     print "-h or --help will print help message"

# class CMLParser:
#     def __init__(self):
#         self.__screenshotDescriptionFileName=""
#         self.customScreenshotDirectoryName=""
#         self.__fileName=""
#     def getSimulationFileName(self):
#         return self.__fileName
#         
#     def processCommandLineOptions(self):
#         import getopt
#         self.__screenshotDescriptionFileName=""
#         self.customScreenshotDirectoryName=""
#         self.outputFrequency=1
#         self.outputFileCoreName="Step"
#         self.exitWhenDone=False
#         self.maxNumberOfRuns=-1
#         self.push_address = None
#         self.return_value_tag = 'generic_label'
#         startSimulation=False
#         
#         opts=None
#         args=None
#         try:
#             opts, args = getopt.getopt(sys.argv[1:], "i:s:o:f:p:l:c:h", ["help","noOutput","exitWhenDone","currentDir=","outputFrequency=","pushAddress=","returnValueTag","maxNumberOfRuns="])
#             print "opts=",opts
#             print "args=",args
#         except getopt.GetoptError, err:
#             # print help information and exit:
#             print str(err) # will print something like "option -a not recognized"
#             # self.usage()
#             sys.exit(1)
#         output = None
#         verbose = False
#         currentDir=""
#         for o, a in opts:
#             print "o=",o
#             print "a=",a
#             if o in ("-i"):
#                 self.__fileName=a
#                 startSimulation=True
#                 
#             elif o in ("-h", "--help"):
#                 self.usage()
#                 return True  # help only
#             elif o in ("-s"):
#                 self.__screenshotDescriptionFileName=a
#             elif o in ("-c"):
#                 self.outputFileCoreName=a                
#             elif o in ("-o"):    
#                 self.customScreenshotDirectoryName=a
#                 self.__noOutput=False
#             elif o in ("--noOutput"):             
#                 self.__noOutput=True
#                 self.outputFrequency=0                
#             elif o in ("-f","--outputFrequency"):             
#                 self.outputFrequency=int(a)
# 
#             elif o in ("-p","--pushAddress"):
#                 self.push_address = str(a)
#                 # print 'GOT pushAddress = ',self.push_address
# 
#             elif o in ("-l","--returnValueTag"):
#                 self.return_value_tag = str(a)
#                 print 'GOT return_value_tag = ',self.return_value_tag
# 
#             elif o in ("--currentDir"):
#                 currentDir=a
#                 print "currentDirectory=",currentDir
#             elif o in ("--exitWhenDone"):
#                 self.exitWhenDone=True                    
#             elif o in ("--maxNumberOfRuns"):
#                 self.maxNumberOfRuns=int(a)
#                 
#             
#             # elif o in ("--exitWhenDone"):             
#                 # self.closePlayerAfterSimulationDone=True 
#                 
#             else:
#                 assert False, "unhandled option"
#                 
#         return False
#                 
#                 
#     def usage(self):
#         print "USAGE: ./runScript.sh -i <simulation file>  -c <outputFileCoreName> "
#         print "-o <customVtkDirectoryName>  -f,--frequency=<frequency> "
#         print "--noOutput will ensure that no output is stored"
#         print "-h or --help will print help message"
