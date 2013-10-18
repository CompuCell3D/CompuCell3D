from __future__ import with_statement
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
    platform=sys.platform
    if platform=='win32':
        sys.path.insert(0,environ["PYTHON_DEPS_PATH"])
    #    else:
    #        swig_path_list=string.split(environ["VTKPATH"])
    #        for swig_path in swig_path_list:
    #            sys.path.append(swig_path)

    # print "PATH=",sys.path
    
setVTKPaths()
# print "PATH=",sys.path  


import os          
import sys
import Configuration
import time

""" Have to import vtk from command line script to make sure vtk output works"""


cc3dSimulationDataHandler=None



def prepareParameterScan(_cc3dSimulationDataHandler):
    '''This fcn returns True if preparation of the next PS run was succesfull or False otherwise - this will usually happen when parameter scan reaches max iteration . 
    '''
    # psu = _cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.psu # parameter scan utils

    # print 'pscanFile=',_cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path
    
    pScanFilePath = _cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path # parameter scan file path
    psu = _cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.psu #parameter scan utils
    
    # We use separate ParameterScanUtils object to handle parameter scan 
    from ParameterScanUtils import ParameterScanUtils
    
    psu=ParameterScanUtils()
    psu.readParameterScanSpecs(pScanFilePath)
    # print 'INITIAL READ O FPARAMETER SCAN------------------------------'
    # print psu.getParameterScanSpecsXMLString()
    # print 'END OF INITIAL READ O FPARAMETER SCAN------------------------------'
    
    paramScanSpecsDirName=os.path.dirname(pScanFilePath)
    
    # # # lockFilePath=os.path.join(paramScanSpecsDirName,'.paramScanLock')
    # # # if not os.path.isfile(lockFilePath):
        # # # open(lockFilePath, 'a').close()
    
    # # reading and writing parameter scan specs to ensure ordering of the file is consisten with internal ordering of the psu 
    # psu.normalizeParameterScanSpecs(pScanFilePath)
    
    
    outputDir = str(Configuration.getSetting('OutputLocation'))

    # # # from FileLock import FileLock
    # fLock=FileLock(file_name=pScanFilePath, timeout=15, delay=0.5)    
    # # fLock=FileLock(file_name=lockFilePath, timeout=10, delay=0.05)    
    
    # # # print 'waiting for lock'
    # # # fLock.acquire()
    # # # print 'lock acquired'
    
    # # time.sleep(1)
    # import time
    # time.sleep(10)
    
    # # # print 'prepareOUTPUT DIR'
    
    # IMPORTANT : this call synchronizes the content of the psu object with the content of the param scan spec file
    # we have to call it inside locked region to avoid possibility that outside locked region two or more cc3d simulations read the same content of the file and thus will attempt to create the same output dir 
    # only one of them will succeed , all others will stop running . The simulation will finish but it will be much slower because only one process will remain
    # # # psu.refreshParamSpecsContent(pScanFilePath) 
    
    customOutputPath=psu.prepareParameterScanOutputDirs(_outputDirRoot=outputDir)
    # # # print 'AFTER prepareOUTPUT DIR, ',customOutputPath
    # # # time.sleep(1)
    
    if not customOutputPath:
        # # # fLock.release()
        # # # print 'RELEASING THE LOCK'
        # # # time.sleep(1)
        return False,False
        
    # # # print 'customOutputPath=',customOutputPath    
    # # # time.sleep(1)    
    
    _cc3dSimulationDataHandler.copySimulationDataFiles(customOutputPath) 


    
    # tweak simulation files according to parameter scan file
    
    #construct path to the just-copied .cc3d file
    cc3dFileBaseName=os.path.basename(_cc3dSimulationDataHandler.cc3dSimulationData.path)
    cc3dFileFullName=os.path.join(customOutputPath,cc3dFileBaseName)
    
    # set output directory for parameter scan
    setSimulationResultStorageDirectory(customOutputPath)
    
    # replace values simulation files with values defined in the  the parameter scan spcs
    psu.replaceValuesInSimulationFiles(_pScanFileName = pScanFilePath, _simulationDir = customOutputPath)    
    
    # save parameter Scan spec file with incremented ityeration
    psu.saveParameterScanState(_pScanFileName = pScanFilePath)
    
    
    return customOutputPath,cc3dFileFullName
    # # # loadCC3DFile(fileName = cc3dFileFullName , forceSingleRun = True)
    
    # # # # # # fLock.release()
    
    
    # # # return True
    
    
    

    


def prepareSingleRun(_cc3dSimulationDataHandler):
    
    import CompuCellSetup
    
    CompuCellSetup.simulationPaths.setSimulationBasePath(_cc3dSimulationDataHandler.cc3dSimulationData.basePath)

    if _cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":       
    
        
        CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(_cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
        
        if _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript!="":
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(_cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)

    
    elif _cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":
    
        
        CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(_cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)
    
def setSimulationResultStorageDirectory(_dir=''):
    if _dir != '':    
        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectory(_dir,False)
    else:
        outputDir = str(Configuration.getSetting('OutputLocation'))                                    
        simulationResultsDirectoryName , baseFileNameForDirectory = CompuCellSetup.getNameForSimDir(fileName,outputDir)
        
        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectoryDirect(simulationResultsDirectoryName)        
        print 'simulationResultsDirectoryName , baseFileNameForDirectory=',(simulationResultsDirectoryName , baseFileNameForDirectory)
        
 
 
    
 
def readCC3DFile(fileName):
    import CompuCellSetup
    import CC3DSimulationDataHandler as CC3DSimulationDataHandler
    
    cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.readCC3DFileFormat(fileName)
    print cc3dSimulationDataHandler.cc3dSimulationData        
    
    
    return cc3dSimulationDataHandler

    
def loadCC3DFile(fileName,forceSingleRun=False):    

    from FileLock import FileLock    
    fLock=FileLock(file_name=fileName, timeout=10, delay=0.05) 
    fLock.acquire()    

    import CompuCellSetup
    import CC3DSimulationDataHandler as CC3DSimulationDataHandler
    
    cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.readCC3DFileFormat(fileName)
    print cc3dSimulationDataHandler.cc3dSimulationData        
    
    
    if   forceSingleRun: # forces loadCC3D to behave as if it was running plane simulation without addons such as e.g. parameter scan
        prepareSingleRun(cc3dSimulationDataHandler)
    else:
        if cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource:
            preparationSuccessful=prepareParameterScan(cc3dSimulationDataHandler)
            if not preparationSuccessful:
                raise AssertionError('Parameter Scan Complete')
            
        else:
            prepareSingleRun(cc3dSimulationDataHandler)
                
    fLock.release()                
    return cc3dSimulationDataHandler         



import vtk     

from os import environ
python_module_path=os.environ["PYTHON_MODULE_PATH"]
appended=sys.path.count(python_module_path)
if not appended:
    sys.path.append(python_module_path)

swig_lib_install_path=os.environ["SWIG_LIB_INSTALL_DIR"]
appended=sys.path.count(swig_lib_install_path)
if not appended:
    sys.path.append(swig_lib_install_path)
        
# sys.path.append(environ["PYTHON_MODULE_PATH"])
# sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

versionStr='3.6.0'    
revisionStr='0'

try:
    import Version            
    versionStr=Version.getVersionAsString()    
    revisionStr=Version.getSVNRevisionAsString()    
except ImportError,e:
    pass

print "CompuCell3D Version: %s Revision: %s\n"%(versionStr,revisionStr)     
    
import CompuCellSetup
from CMLParser import CMLParser

from xml.parsers.expat import ExpatError

try:
    
    from xml.parsers.expat import ExpatError
 
    import re
    from os import environ
    import string
    import traceback
    CompuCellSetup.playerType="CML"

    cmlParser=CompuCellSetup.cmlParser
    
    singleSimulation=True
    
    sim,simthread=None,None
    
    helpOnly = cmlParser.processCommandLineOptions()
    if helpOnly: 
        raise NameError('HelpOnly')
    fileName=cmlParser.getSimulationFileName()
    
    while(True): # by default we are looping the simulation to make sure parameter scans are handled properly
    
        cc3dSimulationDataHandler=None
        
        from FileLock import FileLock    
        with FileLock(file_name=fileName, timeout=10, delay=0.05)  as flock:
        # fLock=FileLock(file_name=fileName, timeout=10, delay=0.05) 
        # fLock.acquire()    
    
            CompuCellSetup.resetGlobals()   
            CompuCellSetup.simulationPaths = CompuCellSetup.SimulationPaths()    
            
            
            sim,simthread = CompuCellSetup.getCoreSimulationObjects(True)
            # CompuCellSetup.cmlFieldHandler.outputFrequency=cmlParser.outputFrequency          

         
            
            
            
            
            
            CompuCellSetup.simulationFileName=fileName
            
                
            setSimulationResultStorageDirectory(cmlParser.customScreenshotDirectoryName) # set Simulation output dir - it can be reset later - at this point only directory name is set. directory gets created later
            
            if re.match(".*\.xml$", fileName): # If filename ends with .xml
                print "GOT FILE ",fileName
                    
                pythonScriptName = CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)
                CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(fileName)
                
                if pythonScriptName!="":
                    CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(pythonScriptName) 
            elif re.match(".*\.py$", fileName):     
                # NOTE: extracting of xml file name from python script is done during script run time so we cannot use CompuCellSetup.simulationPaths.setXmlFileNameFromPython function here
                CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(fileName)

            elif re.match(".*\.cc3d$", fileName):     
                # cc3dSimulationDataHandler=loadCC3DFile(fileName,False)           
                cc3dSimulationDataHandler=readCC3DFile(fileName)
                
             
            
            if cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource:
                singleSimulation = False       
                
                customOutputPath,cc3dFileFullName=prepareParameterScan(cc3dSimulationDataHandler)
                print 'customOutputPath=',customOutputPath
                
                if not cc3dFileFullName:
                    raise AssertionError('Parameter Scan Complete') 
                else:
                    
                    cc3dSimulationDataHandler=readCC3DFile(cc3dFileFullName)
                    
                    prepareSingleRun(cc3dSimulationDataHandler)
                    
            else:
                prepareSingleRun(cc3dSimulationDataHandler)
                
                
            # # # fLock.release()
            
        
        import time
        from random import random
        # time.sleep(5*random())

        # for single run simulation we copy simulation files to the output directory
        if  singleSimulation:            
            cc3dSimulationDataHandler.copySimulationDataFiles(CompuCellSetup.screenshotDirectoryName) 
            
     
        if CompuCellSetup.simulationPaths.simulationPythonScriptName != "":
            # fileObj=file(CompuCellSetup.simulationPaths.simulationPythonScriptName,"r")
            # exec fileObj
            # fileObj.close()
            execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
        else:
            sim,simthread = CompuCellSetup.getCoreSimulationObjects()
            # CompuCellSetup.cmlFieldHandler.outputFrequency=cmlParser.outputFrequency          
            import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()
            #import CompuCellSetup        
            CompuCellSetup.initializeSimulationObjects(sim,simthread)
            steppableRegistry = CompuCellSetup.getSteppableRegistry()
            CompuCellSetup.mainLoop(sim,simthread,steppableRegistry) # main loop - simulation is invoked inside this function
        
        print 'FINISHED MAIN LOOP'        
        # jumping out of the loop when running single simulation. Will stay in the loop for e.g. parameter scan 
        if singleSimulation:
            break
        

except IndentationError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message = traceback.format_exc()
    print traceback_message
    print "Indentation Error"+e.message
except SyntaxError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message = traceback.format_exc()
    print traceback_message
    print "Syntax Error"+e.message
except IOError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message = traceback.format_exc()
    print traceback_message
    print "IOerror Error"+e.message
except ImportError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message = traceback.format_exc()
    print traceback_message
    print "import Error"+e.message
except ExpatError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    xmlFileName = CompuCellSetup.simulationPaths.simulationXMLFileName
    print "Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message


except AssertionError,e:
    if CompuCellSetup.simulationObjectsCreated:        
        sim.finish()        
    print "Assertion Error: ",e.message
    
except CompuCellSetup.CC3DCPlusPlusError,e:
    print "RUNTIME ERROR IN C++ CODE: ",e.message
except NameError,e:
    pass
except:
    print 'GENERAL EXCEPTION \n\n\n\n'
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    if helpOnly:
        raise
    else:
        traceback_message = traceback.format_exc()
        print "Unexpected Error:",traceback_message

