
import ParameterScanEnums
import ErrorCodes

# enabling with statement in python 2.5

'''
SWIG LIBRARY LOADING ORDER: Before attempting to load swig libray it is necessary to first call 
   sim,simthread = CompuCellSetup.getCoreSimulationObjects(True)
   loadCC3DFile imports XMLUtils (swig library) and if getCoreSimulationObjects is not called before importting e.g. XMLUtils then segfault may appear sometime during run
   The likely causes are:
   1) Using execfile to run actual simulation from this python script
   2) Using global variables in CompuCellsetup
   3) order in which swig libraries are loaded matters.
   
   After we moved getCoreSimulationObjects to be executed as early as possible most of the segfault erros disappeared - those errors were all associated with SwigPyIterator 
   when we tried to iterate over CC3D C++ STL based containers - e.g. sets, maps etc. using iterators provided byt swig wrappers like iter() , itervalues(), iterators()
   Hand-written iterators were OK tohugh. The segfaults appearedonly in the command line runs i.e. without the player
'''


def setVTKPaths():
    import sys
    from os import environ
    import string
    import sys
    platform = sys.platform
    if platform == 'win32':
        sys.path.insert(0, environ["PYTHON_DEPS_PATH"])
        #    else:
        #        swig_path_list=string.split(environ["VTKPATH"])
        #        for swig_path in swig_path_list:
        #            sys.path.append(swig_path)

        # print "PATH=",sys.path


setVTKPaths()
# print "PATH=",sys.path

import os, sys
from os import environ


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
    sys.exit(1)

# enable it during debugging in pycharm
sys.excepthook = except_hook

python_module_path = os.environ["PYTHON_MODULE_PATH"]
appended = sys.path.count(python_module_path)
if not appended:
    sys.path.append(python_module_path)

swig_lib_install_path = os.environ["SWIG_LIB_INSTALL_DIR"]
appended = sys.path.count(swig_lib_install_path)
if not appended:
    sys.path.append(swig_lib_install_path)

# import Configuration # we do not import Configuration in order to avoid PyQt4 dependency for runScript
import time

""" Have to import vtk from command line script to make sure vtk output works"""

cc3dSimulationDataHandler = None



def getRootOutputDir():
    '''returns output location stored in the Configuration's 'OutputLocation' entry.
      if it cannot import Configuration it returns os.path.join(os.path.expanduser('~'),'CC3DWorkspace')
    '''
    try:
        from . import Configuration
        outputDir = str(Configuration.getSetting('OutputLocation'))
    except:
        return os.path.abspath(os.path.join(os.path.expanduser('~'), 'CC3DWorkspace'))

    return outputDir


def prepareParameterScan(_cc3dSimulationDataHandler):
    '''This fcn returns True if preparation of the next PS run was succesfull or False otherwise - this will usually happen when parameter scan reaches max iteration . 
    '''

    pScanFilePath = _cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path  # parameter scan file path
    cc3dProjectPath = _cc3dSimulationDataHandler.cc3dSimulationData.path
    cc3dProjectDir = _cc3dSimulationDataHandler.cc3dSimulationData.basePath

    # psu = _cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.psu #parameter scan utils

    # checking if simulation file directory is writeable if not parameterscan cannot run properly - writeable simulation fiel directory is requirement for parameter scan
    if not os.access(cc3dProjectDir, os.W_OK):
        #         raise AssertionError('parameter Scan Error: CC3D project directory:'+cc3dProjectDir+' has to be writeable. Please change permission on the directory of the .cc3d project')
        raise AssertionError('Parameter Scan ERRORCODE=' + str(
            ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE) + ': : CC3D project directory:' + cc3dProjectDir + ' has to be writeable. Please change permission on the directory of the .cc3d project')
        # check if parameter scan file is writeable
    if not os.access(pScanFilePath, os.W_OK):
        raise AssertionError('Parameter Scan ERRORCODE=' + str(
            ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE) + ': Parameter Scan xml file :' + pScanFilePath + ' has to be writeable. Please change permission on this file')
    # raise AssertionError('parameter Scan Error: Parameter Scan xml file :'+pScanFilePath+ ' has to be writeable. Please change permission on this file')

    # We use separate ParameterScanUtils object to handle parameter scan 
    from ParameterScanUtils import ParameterScanUtils

    psu = ParameterScanUtils()
    psu.readParameterScanSpecs(pScanFilePath)

    paramScanSpecsDirName = os.path.dirname(pScanFilePath)

    outputDir = getRootOutputDir()  # output dir is determined in a way that dpes not require PyQt4  and Configuration module

    customOutputPath = psu.prepareParameterScanOutputDirs(_outputDirRoot=outputDir)

    if not customOutputPath:
        return False, False

    _cc3dSimulationDataHandler.copy_simulation_data_files(customOutputPath)

    # tweak simulation files according to parameter scan file

    # construct path to the just-copied .cc3d file
    cc3dFileBaseName = os.path.basename(_cc3dSimulationDataHandler.cc3dSimulationData.path)
    cc3dFileFullName = os.path.join(customOutputPath, cc3dFileBaseName)

    # set output directory for parameter scan
    setSimulationResultStorageDirectory(customOutputPath)

    # replace values simulation files with values defined in the  the parameter scan spcs
    psu.replaceValuesInSimulationFiles(_pScanFileName=pScanFilePath, _simulationDir=customOutputPath)

    # save parameter Scan spec file with incremented ityeration
    psu.saveParameterScanState(_pScanFileName=pScanFilePath)

    return customOutputPath, cc3dFileFullName


def prepareSingleRun(_cc3dSimulationDataHandler):
    import CompuCellSetup

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


def setSimulationResultStorageDirectory(_dir=''):
    if _dir != '' and _dir != None:
        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectory(_dir, False)
    else:
        outputDir = getRootOutputDir()
        simulationResultsDirectoryName, baseFileNameForDirectory = CompuCellSetup.getNameForSimDir(fileName, outputDir)

        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectoryDirect(simulationResultsDirectoryName)


def readCC3DFile(fileName):
    import CompuCellSetup
    import CC3DSimulationDataHandler as CC3DSimulationDataHandler

    cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.read_cc3_d_file_format(fileName)
    print(cc3dSimulationDataHandler.cc3dSimulationData)

    return cc3dSimulationDataHandler


def loadCC3DFile(fileName, forceSingleRun=False):
    from FileLock import FileLock
    fLock = FileLock(file_name=fileName, timeout=10, delay=0.05)
    fLock.acquire()

    import CompuCellSetup
    import CC3DSimulationDataHandler as CC3DSimulationDataHandler

    cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.read_cc3_d_file_format(fileName)
    print(cc3dSimulationDataHandler.cc3dSimulationData)

    if forceSingleRun:  # forces loadCC3D to behave as if it was running plane simulation without addons such as e.g. parameter scan
        prepareSingleRun(cc3dSimulationDataHandler)
    else:
        if cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource:
            preparationSuccessful = prepareParameterScan(cc3dSimulationDataHandler)
            if not preparationSuccessful:
                #                 raise AssertionError('Parameter Scan Complete')
                raise AssertionError('Parameter Scan ERRORCODE=' + str(
                    ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE) + ': Parameter Scan Complete')

        else:
            prepareSingleRun(cc3dSimulationDataHandler)

    fLock.release()
    return cc3dSimulationDataHandler


import vtk

# sys.path.append(environ["PYTHON_MODULE_PATH"])
# sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

versionStr = '3.6.0'
revisionStr = '0'

try:
    import Version

    versionStr = Version.getVersionAsString()
    revisionStr = Version.getSVNRevisionAsString()
except ImportError as e:
    pass

print("CompuCell3D Version: %s Revision: %s\n" % (versionStr, revisionStr))

import CompuCellSetup
from CMLParser import CMLParser

from xml.parsers.expat import ExpatError



try:

    from xml.parsers.expat import ExpatError

    import re
    from os import environ
    import string
    import traceback
    import time

    CompuCellSetup.playerType = "CML"

    cmlParser = CompuCellSetup.cmlParser

    singleSimulation = True

    relaunch = False
    allowRelaunch = True

    sim, simthread = None, None
    helpOnly = cmlParser.parse_cml()
    cml_args = cmlParser.cml_args

    # print 'BEFORE cmlParser.processCommandLineOptions() \n\n\n\n'
    # helpOnly = cmlParser.processCommandLineOptions()
    # print 'GOT PAST cmlParser.processCommandLineOptions() \n\n\n\n'
    #
    # if helpOnly:
    #     raise NameError('HelpOnly')

    # setting up push address


    if hasattr(cmlParser, 'push_address'):
        CompuCellSetup.set_push_address(cmlParser.push_address)


    # setting up return tag
    if hasattr(cmlParser, 'return_value_tag'):
        CompuCellSetup.set_return_value_tag(cmlParser.return_value_tag)

    fileName = cmlParser.getSimulationFileName()

    consecutiveRunCounter = 0
    maxNumberOfConsecutiveRuns = 10
    # extracting from the runScript maximum number of consecutive runs
    try:
        maxNumberOfConsecutiveRuns = int(os.environ["MAX_NUMBER_OF_CONSECUTIVE_RUNS"])
        if cml_args.maxNumberOfConsecutiveRuns >0:
            maxNumberOfConsecutiveRuns = cmlParser.maxNumberOfRuns
        # if cmlParser.maxNumberOfRuns > 0:
        #     maxNumberOfConsecutiveRuns = cmlParser.maxNumberOfRuns

        # we reset max number of consecutive runs to 1 because we want each simulation in parameter scan
        # initiated by the psrun.py script to be an independent run after which runScript terminatesrestarts again for the next run
        if cml_args.exitWhenDone:
            maxNumberOfConsecutiveRuns = 1
            allowRelaunch = False
    except:  # if for whatever reason we cannot do it we stay with the default value
        pass

    while (True):  # by default we are looping the simulation to make sure parameter scans are handled properly

        cc3dSimulationDataHandler = None

        from FileLock import FileLock

        with FileLock(file_name=fileName, timeout=10, delay=0.05)  as flock:

            CompuCellSetup.resetGlobals()
            CompuCellSetup.simulationPaths = CompuCellSetup.SimulationPaths()

            sim, simthread = CompuCellSetup.getCoreSimulationObjects(True)

            # import PlayerPython
            #
            # field_handler = simthread
            #
            # field_extractor_local = PlayerPython.FieldExtractor()
            # field_extractor_local.setFieldStorage(field_handler.fieldStorage)
            # field_extractor_local.init(sim)

            setSimulationResultStorageDirectory(
                cmlParser.customScreenshotDirectoryName)  # set Simulation output dir - it can be reset later - at this point only directory name is set. directory gets created later

            CompuCellSetup.simulationFileName = fileName

            # print 'GOT HERE'
            if re.match(".*\.xml$", fileName):  # If filename ends with .xml
                print("GOT FILE ", fileName)

                pythonScriptName = CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)
                CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(fileName)

                if pythonScriptName != "":
                    CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(pythonScriptName)
            elif re.match(".*\.py$", fileName):
                # NOTE: extracting of xml file name from python script is done during script run time so we cannot use CompuCellSetup.simulationPaths.setXmlFileNameFromPython function here
                CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(fileName)

            elif re.match(".*\.cc3d$", fileName):
                # cc3dSimulationDataHandler=loadCC3DFile(fileName,False)           
                cc3dSimulationDataHandler = readCC3DFile(fileName)
                import Version

                currentVersion = Version.getVersionAsString()
                currentVersionInt = currentVersion.replace('.', '')
                projectVersion = cc3dSimulationDataHandler.cc3dSimulationData.version
                projectVersionInt = projectVersion.replace('.', '')
                if int(projectVersionInt) > int(currentVersionInt):
                    print('\n\n\n--------------- COMPUCELL3D VERSION MISMATCH\n\n')
                    print('Your CompuCell3D version %s might be too old for the project you are trying to run.\n The least version project requires is %s. \n You may run project at your own risk' % (
                        currentVersion, projectVersion))
                    import time

                    time.sleep(5)
            else:
                raise RuntimeError("Invalid simulation file: %s "%fileName)
            if cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource:
                singleSimulation = False

                customOutputPath, cc3dFileFullName = prepareParameterScan(cc3dSimulationDataHandler)
                print('customOutputPath=', customOutputPath)

                if not cc3dFileFullName:
                    raise AssertionError('Parameter Scan Complete')
                else:

                    cc3dSimulationDataHandler = readCC3DFile(cc3dFileFullName)

                    prepareSingleRun(cc3dSimulationDataHandler)

            else:

                prepareSingleRun(cc3dSimulationDataHandler)


                # # # fLock.release()

        # for single run simulation we copy simulation files to the output directory
        if singleSimulation:
            CompuCellSetup.cc3dSimulationDataHandler = cc3dSimulationDataHandler
            cc3dSimulationDataHandler.copySimulationDataFiles(CompuCellSetup.screenshotDirectoryName)

        if CompuCellSetup.simulationPaths.simulationPythonScriptName != "":
            exec(compile(open(CompuCellSetup.simulationPaths.simulationPythonScriptName).read(), CompuCellSetup.simulationPaths.simulationPythonScriptName, 'exec'))
        else:
            sim, simthread = CompuCellSetup.getCoreSimulationObjects()
            # CompuCellSetup.cmlFieldHandler.outputFrequency=cmlParser.outputFrequency          
            import \
                CompuCell  # notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

            # import CompuCellSetup
            CompuCellSetup.initializeSimulationObjects(sim, simthread)
            steppableRegistry = CompuCellSetup.getSteppableRegistry()
            CompuCellSetup.mainLoop(sim, simthread,
                                    steppableRegistry)  # main loop - simulation is invoked inside this function
            sim.cleanAfterSimulation()
            sim = None

        print('FINISHED MAIN LOOP')
        #         if not allowRelanuch:
        #             break

        # jumping out of the loop when running single simulation. Will stay in the loop for e.g. parameter scan 
        if singleSimulation:
            break
        else:
            consecutiveRunCounter += 1
            if consecutiveRunCounter >= maxNumberOfConsecutiveRuns:
                relaunch = True
                break

                #     print 'allowRelaunch=',allowRelaunch,' relaunch=',relaunch

    if allowRelaunch and relaunch:
        from ParameterScanUtils import getParameterScanCommandLineArgList
        from SystemUtils import getCC3DRunScriptPath

        popenArgs = [getCC3DRunScriptPath()] + getParameterScanCommandLineArgList(fileName)
        # print 'popenArgs=',popenArgs

        #         print 'WILL RESTART RUN SCRIPT FOR PARAMETER SCAN'
        #         import time
        #         time.sleep(5)

        from subprocess import Popen

        cc3dProcess = Popen(popenArgs)



except (IndentationError, SyntaxError, IOError, ImportError, NameError) as e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    # traceback_message = traceback.format_exc()

    # print  e.__class__.__name__
    print('CC3D encountered ' + e.__class__.__name__ + ' : ' + e.message)
    traceback_message = traceback.format_exc()
    print(traceback_message)

    sys.exit(ErrorCodes.EXCEPTION_IN_CC3D)

except ExpatError as e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    xmlFileName = CompuCellSetup.simulationPaths.simulationXMLFileName
    print("Error in XML File", "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message)
    sys.exit(ErrorCodes.EXCEPTION_IN_CC3D)

except AssertionError as e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    print("Assertion Error: ", e.message)

    if e.message.startswith('Parameter Scan ERRORCODE=' + str(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)):
        sys.exit(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)

    sys.exit(ErrorCodes.EXCEPTION_IN_CC3D)


except CompuCellSetup.CC3DCPlusPlusError as e:
    print("RUNTIME ERROR IN C++ CODE: ", e.message)
    sys.exit(ErrorCodes.EXCEPTION_IN_CC3D)
# except NameError,e:
#     pass
except:
    #     print 'GENERAL EXCEPTION \n\n\n\n'
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    if helpOnly:
        raise
    else:
        traceback_message = traceback.format_exc()
        print("Unexpected Error:", traceback_message)
    sys.exit(ErrorCodes.EXCEPTION_IN_CC3D)
