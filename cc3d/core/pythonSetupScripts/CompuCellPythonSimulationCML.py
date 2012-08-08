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

""" Have to import vtk from command line script to make sure vtk output works"""

cc3dSimulationDataHandler=None

def loadCC3DFile(fileName):
    import CompuCellSetup
    import CC3DSimulationDataHandler as CC3DSimulationDataHandler
    
    cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler(None)
    cc3dSimulationDataHandler.readCC3DFileFormat(fileName)
    print cc3dSimulationDataHandler.cc3dSimulationData        
    
    
    CompuCellSetup.simulationPaths.setSimulationBasePath(cc3dSimulationDataHandler.cc3dSimulationData.basePath)

    if cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":       
    
        
        CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
        
        if cc3dSimulationDataHandler.cc3dSimulationData.xmlScript!="":
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)

    
    elif cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":
    
        
        CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)




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
      
try:
    from xml.parsers.expat import ExpatError
 
    import re
    from os import environ
    import string
    import traceback
    CompuCellSetup.playerType="CML"

    cmlParser=CompuCellSetup.cmlParser
    # CMLParser()
    helpOnly = cmlParser.processCommandLineOptions()
    if helpOnly: 
        raise NameError('HelpOnly')
    
    CompuCellSetup.simulationPaths = CompuCellSetup.SimulationPaths()    
    
    
    sim,simthread = CompuCellSetup.getCoreSimulationObjects(True)
    # CompuCellSetup.cmlFieldHandler.outputFrequency=cmlParser.outputFrequency          

 # will set a directory where one may want to save simulation results
    if cmlParser.customScreenshotDirectoryName != "":
        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectory(cmlParser.customScreenshotDirectoryName,False)
 
 
    fileName=cmlParser.getSimulationFileName()
    
    CompuCellSetup.simulationFileName=fileName
    
    # will set a directory where one may want to save simulation results
    if cmlParser.customScreenshotDirectoryName=="":
        # CompuCellSetup.simulationPaths.setSimulationResultStorageDirectory(fileName,True)
        # simulationResultsDirectoryName , baseFileNameForDirectory=CompuCellSetup.makeSimDir(fileName)
        outputDir = str(Configuration.getSetting("OutputLocation"))                                    
        simulationResultsDirectoryName , baseFileNameForDirectory = CompuCellSetup.getNameForSimDir(fileName,outputDir)
        
        CompuCellSetup.simulationPaths.setSimulationResultStorageDirectoryDirect(simulationResultsDirectoryName)        
        print "simulationResultsDirectoryName , baseFileNameForDirectory=",(simulationResultsDirectoryName , baseFileNameForDirectory)
        
 
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
        loadCC3DFile(fileName)
    

 
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
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    if helpOnly:
        raise
    else:
        traceback_message = traceback.format_exc()
        print "Unexpected Error:",traceback_message

