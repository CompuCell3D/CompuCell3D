import xml.etree.ElementTree as ET
import os
import ProjectFileStore

class CC3DReader:

    def readCC3DFile(self, cc3dFilePath):
        #TO-DO: asserts for file path
        tree = ET.parse(cc3dFilePath)
        simulationRoot = tree.getroot()
        version = simulationRoot.get('version')

        '''
        Read the PythonScript tag which holds file path for python script for the simulation
        '''
        pythonScriptElementList  = simulationRoot.findall('PythonScript')
        if pythonScriptElementList is None or len(pythonScriptElementList) != 1:
            print "Invalid number of 'PythonScript' elements"
            # raise error
            return

        pythonScriptElement = pythonScriptElementList[0]
        pythonScriptPath = pythonScriptElement.text
        projectDirectoryPath = os.path.dirname(os.path.abspath(cc3dFilePath))
        pythonScriptPath = os.path.join(projectDirectoryPath, pythonScriptPath)
        ProjectFileStore.pythonScriptPath = pythonScriptPath

        '''
        '''
        xmlScriptElementList = simulationRoot.findall('XMLScript')
        if xmlScriptElementList is not None and len(xmlScriptElementList) != 0:
            print "This project contents XML Script. Please use another CompuCell3D invocation method."
            # raise error