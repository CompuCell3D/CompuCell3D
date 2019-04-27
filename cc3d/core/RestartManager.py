# -*- coding: utf-8 -*-
import os, sys
import re
import pickle
from cc3d.cpp import CompuCell
from cc3d.cpp import SerializerDEPy
from cc3d.core.PySteppables import CellList
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.core import Version
from cc3d import CompuCellSetup
from pathlib import Path

import warnings


def _pickleVector3(_vec):
    return CompuCell.Vector3, (_vec.fX, _vec.fY, _vec.fZ)


import copyreg

copyreg.pickle(CompuCell.Vector3, _pickleVector3)


class RestartManager:

    def __init__(self, _sim=None):

        sim = CompuCellSetup.persistent_globals.simulator
        self.serializer = SerializerDEPy.SerializerDE()

        self.serializer.init(sim)

        self.cc3dSimOutputDir = ''
        self.serializeDataList = []
        self.__step_number_of_digits = 0  # field size for formatting step number output
        self.__completedRestartOutputPath = ''
        self.__allowMultipleRestartDirectories = True
        self.__outputFrequency = 0
        self.__baseSimulationFilesCopied = False

        # variables used during restarting
        self.__restartDirectory = ''
        self.__restartFile = ''
        self.__restartVersion = 0
        self.__restartBuild = 0
        self.__restartStep = 0
        self.__restartResourceDict = {}

        self.cc3dSimulationDataHandler = None

    def getRestartStep(self):
        return self.__restartStep

    def prepareRestarter(self):
        """
        Performs basic setup before attempting a restart
        :return: None
        """

        # todo - fix
        self.__allowMultipleRestartDirectories = False
        self.__outputFrequency = 1

        return

        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            from . import CC3DSimulationDataHandler
            cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))

            # checking is serializer resource exists
            if cc3dSimulationDataHandler.cc3dSimulationData.serializerResource:
                self.__allowMultipleRestartDirectories = cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.allowMultipleRestartDirectories
                self.__outputFrequency = cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.outputFrequency

    def restartEnabled(self):
        """
        reads .cc3d project file and checks if restart is enabled
        :return: {bool}
        """
        return False

        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            print("EXTRACTING restartEnabled")

            from . import CC3DSimulationDataHandler

            cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))

            return cc3dSimulationDataHandler.cc3dSimulationData.restartEnabled()

        return False

    def appendXMLStub(selt, _rootElem, _sd):
        """
        Internal function in the restart manager - manipulatex xml file that describes the layout of
        restart files
        :param _rootElem: {instance of CC3DXMLElement}
        :param _sd: {object that has basic information about serialized module}
        :return: None
        """
        baseFileName = os.path.basename(_sd.fileName)
        attributeDict = {"ModuleName": _sd.moduleName, "ModuleType": _sd.moduleType, "ObjectName": _sd.objectName,
                         "ObjectType": _sd.objectType, "FileName": baseFileName, 'FileFormat': _sd.fileFormat}
        _rootElem.ElementCC3D('ObjectData', attributeDict)

    def getRestartOutputRootPath(self, _restartOutputPath):
        """
        returns path to the  output root directory e.g. <outputFolder>/restart_200
        :param _restartOutputPath: {str}
        :return:{str}
        """
        restartOutputRootPath = os.path.dirname(_restartOutputPath)

        # normalizing path
        restartOutputRootPath = os.path.abspath(restartOutputRootPath)

        return restartOutputRootPath

    def setup_restart_output_directory(self, _step=0):
        """
        Prpares restart directory
        :param _step: {int} Monte Carlo Step
        :return: {None}
        """

        pg = CompuCellSetup.persistent_globals
        output_dir_root = pg.output_directory
        if not self.__step_number_of_digits:
            self.__step_number_of_digits = len(str(pg.simulator.getNumSteps()))

        restart_output_dir = Path(output_dir_root).joinpath('restart_' + str(_step).zfill(self.__step_number_of_digits))
        # restart_output_dir.

        restart_output_dir.mkdir(parents=True, exist_ok=True)

        return str(restart_output_dir)

        print('CompuCellSetup.screenshotDirectoryName=', CompuCellSetup.screenshotDirectoryName)

        self.cc3dSimOutputDir = CompuCellSetup.screenshotDirectoryName

        if not self.__step_number_of_digits:
            self.__step_number_of_digits = len(str(self.sim.getNumSteps()))

        restartOutputPath = ''
        simFilesOutputPath = ''
        if self.cc3dSimOutputDir == '':
            if str(CompuCellSetup.simulationFileName) != '':
                (self.cc3dSimOutputDir, baseScreenshotName) = CompuCellSetup.makeSimDir(
                    str(CompuCellSetup.simulationFileName))
                CompuCellSetup.screenshotDirectoryName = self.cc3dSimOutputDir

                # fills string with 0's up to self.__stepNumberOfDigits
                restartOutputPath = os.path.join(self.cc3dSimOutputDir, 'restart_' + string.zfill(str(_step),
                                                                                                  self.__step_number_of_digits))
                simFilesOutputPath = restartOutputPath

                # one more level of nesting
                restartOutputPath = os.path.join(restartOutputPath,
                                                 'restart')

                try:
                    os.makedirs(restartOutputPath)
                except IOError as e:
                    restartOutputPath = ''

        else:
            self.cc3dSimOutputDir = self.cc3dSimOutputDir

            # fills string with 0's up to self.__stepNumberOfDigits
            restartOutputPath = os.path.join(self.cc3dSimOutputDir,
                                             'restart_' + string.zfill(str(_step), self.__step_number_of_digits))
            simFilesOutputPath = restartOutputPath
            # one more level of nesting
            restartOutputPath = os.path.join(restartOutputPath,
                                             'restart')

            try:
                os.makedirs(restartOutputPath)
            except IOError as e:
                restartOutputPath = ''

        # we only copy simulation files if simulation run in in the .cc3d format                
        import re
        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):

            from . import CC3DSimulationDataHandler

            cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))

            # copying  verbatim simulation files
            if not self.__baseSimulationFilesCopied:
                cc3dSimulationDataHandler.copySimulationDataFiles(self.cc3dSimOutputDir)
                self.__baseSimulationFilesCopied = True

            # copying modified simulation files - with restart modification
            if simFilesOutputPath != '':
                cc3dSimulationDataHandler.copySimulationDataFiles(simFilesOutputPath)
                cc3dSimulationDataHandlerLocal = CC3DSimulationDataHandler.CC3DSimulationDataHandler()

                simBaseName = os.path.basename(str(CompuCellSetup.simulationFileName))
                # path to newly copied simulation file
                simFullName = os.path.join(simFilesOutputPath, simBaseName)
                # read newly copied simulation file - we will add restart tags to it
                cc3dSimulationDataHandlerLocal.readCC3DFileFormat(simFullName)

                print('\n\n\n\n cc3dSimulationDataHandlerLocal.cc3dSimulationData=',
                      cc3dSimulationDataHandlerLocal.cc3dSimulationData)

                # update simulation size in the XML  in case it has changed during the simulation 
                if cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript != '':
                    print('cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript=',
                          cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript)
                    self.updateXMLScript(cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript)
                elif cc3dSimulationDataHandlerLocal.cc3dSimulationData.pythonScript != '':
                    self.updatePythonScript(cc3dSimulationDataHandlerLocal.cc3dSimulationData.pythonScript)

                # if serialize resource exists we only modify it by adding restart simulation element
                if cc3dSimulationDataHandlerLocal.cc3dSimulationData.serializerResource:
                    cc3dSimulationDataHandlerLocal.cc3dSimulationData.serializerResource.restartDirectory = 'restart'
                    cc3dSimulationDataHandlerLocal.writeCC3DFileFormat(simFullName)
                else:  # otherwise we create new simulation resource and add restart simulation element
                    cc3dSimulationDataHandlerLocal.cc3dSimulationData.addNewSerializerResource(_restartDir='restart')
                    cc3dSimulationDataHandlerLocal.writeCC3DFileFormat(simFullName)

            # if self.cc3dSimOutputDir!='':
            # cc3dSimulationDataHandler.copySimulationDataFiles(self.cc3dSimOutputDir)

        return restartOutputPath

    def updatePythonScript(self, _fileName):
        """
        Manipulates Python script - alters the content to make sure it is restart -ready
        :param _fileName: {str} path to Python file
        :return: None
        """
        if _fileName == '':
            return

        import re
        dimRegex = re.compile('([\s\S]*.ElementCC3D\([\s]*"Dimensions")([\S\s]*)(\)[\s\S]*)')
        commentRegex = re.compile('^([\s]*#)')

        try:
            fXMLNew = open(_fileName + '.new', 'w')
        except IOerror as e:
            print(__file__ + ' updatePythonScript: could not open ', _fileName, ' for writing')

        fieldDim = self.sim.getPotts().getCellFieldG().getDim()

        for line in open(_fileName):
            lineTmp = line.rstrip()
            groups = dimRegex.search(lineTmp)

            commentGroups = commentRegex.search(lineTmp)
            if commentGroups:
                print(line.rstrip(), file=fXMLNew)
                continue

            if groups and groups.lastindex == 3:
                dimString = ',{"x":' + str(fieldDim.x) + ',' + '"y":' + str(fieldDim.y) + ',' + '"z":' + str(
                    fieldDim.z) + '}'
                newLine = dimRegex.sub(r'\1' + dimString + r'\3', lineTmp)
                print(newLine, file=fXMLNew)
            else:
                print(line.rstrip(), file=fXMLNew)

        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName + '.new', _fileName)

    def updateXMLScript(self, _fileName=''):

        """
        Manipulates XML script - alters the content to make sure it is restart -ready
        :param _fileName: {str} path to XML file
        :return: None
        """

        if _fileName == '':
            return

        import re
        dimRegex = re.compile('([\s]*<Dimensions)([\S\s]*)(/>[\s]*)')

        try:
            fXMLNew = open(_fileName + '.new', 'w')
        except IOerror as e:
            print(__file__ + ' updateXMLScript: could not open ', _fileName, ' for writing')

        fieldDim = self.sim.getPotts().getCellFieldG().getDim()
        for line in open(_fileName):
            lineTmp = line.rstrip()
            groups = dimRegex.search(lineTmp)

            if groups and groups.lastindex == 3:
                dimString = ' x="' + str(fieldDim.x) + '" ' + 'y="' + str(fieldDim.y) + '" ' + 'z="' + str(
                    fieldDim.z) + '" '
                newLine = dimRegex.sub(r'\1' + dimString + r'\3', lineTmp)
                print(newLine, file=fXMLNew)
            else:

                print(line.rstrip(), file=fXMLNew)

        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName + '.new', _fileName)

    def readRestartFile(self, _fileName):
        """
        reads XML file that holds information about restart data
        :param _fileName: {str}
        :return: None
        """
        from . import XMLUtils
        xml2ObjConverter = XMLUtils.Xml2Obj()

        fileFullPath = os.path.abspath(_fileName)

        root_element = xml2ObjConverter.Parse(fileFullPath)  # this is RestartFiles element
        if root_element.findAttribute('Version'):
            self.__restartVersion = root_element.getAttribute('Version')
        if root_element.findAttribute('Build'):
            self.__restartVersion = root_element.getAttributeAsInt('Build')

        stepElem = root_element.getFirstElement('Step')

        if stepElem:
            self.__restartStep = stepElem.getInt()

        restartObjectElements = XMLUtils.CC3DXMLListPy(root_element.getElements('ObjectData'))

        import SerializerDEPy
        if restartObjectElements:
            for elem in restartObjectElements:
                sd = SerializerDEPy.SerializeData()
                if elem.findAttribute('ObjectName'):
                    sd.objectName = elem.getAttribute('ObjectName')
                if elem.findAttribute('ObjectType'):
                    sd.objectType = elem.getAttribute('ObjectType')
                if elem.findAttribute('ModuleName'):
                    sd.moduleName = elem.getAttribute('ModuleName')
                if elem.findAttribute('ModuleType'):
                    sd.moduleType = elem.getAttribute('ModuleType')
                if elem.findAttribute('FileName'):
                    sd.fileName = elem.getAttribute('FileName')
                if elem.findAttribute('FileFormat'):
                    sd.fileFormat = elem.getAttribute('FileFormat')
                if sd.objectName != '':
                    self.__restartResourceDict[sd.objectName] = sd
        print('self.__restartResourceDict=', self.__restartResourceDict)

    def loadRestartFiles(self):
        """
        Loads restart files
        :return: None
        """

        import CompuCellSetup
        import re

        print("\n\n\n\n REASTART MANAGER CompuCellSetup.simulationFileName=", CompuCellSetup.simulationFileName)

        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):

            print("EXTRACTING restartEnabled")
            from . import CC3DSimulationDataHandler

            cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
            print("cc3dSimulationDataHandler.cc3dSimulationData.serializerResource=",
                  cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory)
            if cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory != '':
                restartFileLocation = os.path.dirname(str(CompuCellSetup.simulationFileName))
                self.__restartDirectory = os.path.join(restartFileLocation, 'restart')
                self.__restartDirectory = os.path.abspath(self.__restartDirectory)  # normalizing path format

                self.__restartFile = os.path.join(self.__restartDirectory, 'restart.xml')
                print('self.__restartDirectory=', self.__restartDirectory)
                print('self.__restartFile=', self.__restartFile)
                self.readRestartFile(self.__restartFile)

        # ---------------------- LOADING RESTART FILES    --------------------
        # loading cell field    

        self.loadCellField()
        # loading concentration fields (scalar fields) from PDE solvers            
        self.loadConcentrationFields()
        # loading extra scalar fields   - used in Python only
        self.loadScalarFields()
        # loading extra scalar fields cell level  - used in Python only
        self.loadScalarFieldsCellLevel()
        # loading extra vector fields  - used in Python only
        self.loadVectorFields()
        # loading extra vector fields cell level  - used in Python only
        self.loadVectorFieldsCellLevel()
        # loading core cell  attributes
        self.loadCoreCellAttributes()
        # load cell Python attributes
        self.loadPythonAttributes()
        # load SBMLSolvers -  free floating SBML Solvers are loaded and initialized and those associated with cell are initialized - they are loaded by  self.loadPythonAttributes
        self.loadSBMLSolvers()
        # load bionetSolver Data
        self.loadBionetSolver()
        # load adhesionFlex plugin
        self.loadAdhesionFlex()
        # load chemotaxis plugin        
        self.loadChemotaxis()
        # load LengthConstraint plugin        
        self.loadLengthConstraint()
        # load ConnectivityGlobal plugin        
        self.loadConnectivityGlobal()
        # load ConnectivityLocalFlex plugin        
        self.loadConnectivityLocalFlex()
        # load FocalPointPlasticity plugin        
        self.loadFocalPointPlasticity()
        # load ContactLocalProduct plugin        
        self.loadContactLocalProduct()
        # load CellOrientation plugin        
        self.loadCellOrientation()
        # load PolarizationVector plugin        
        self.loadPolarizationVector()
        # load loadPolarization23 plugin        
        self.loadPolarization23()

        # load steering panel
        self.loadSteeringPanel()

        # ---------------------- END OF LOADING RESTART FILES    --------------------

    #

    def loadCellField(self, ):
        """
        Restores Cell Field
        :return: None
        """
        import SerializerDEPy
        if 'CellField' in list(self.__restartResourceDict.keys()):
            sd = self.__restartResourceDict['CellField']
            # full path to cell field serialized recource
            fullPath = os.path.join(self.__restartDirectory, sd.fileName)
            fullPath = os.path.abspath(fullPath)  # normalizing path format
            tmpFileName = sd.fileName
            sd.fileName = fullPath
            self.serializer.loadCellField(sd)
            sd.fileName = tmpFileName

    def loadConcentrationFields(self):
        """
        restores chemical fields
        :return: None
        """

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectType == 'ConcentrationField':
                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                tmpFileName = sd.fileName
                sd.fileName = fullPath
                self.serializer.loadConcentrationField(sd)
                sd.fileName = tmpFileName

    def loadScalarFields(self):
        """
        restores user-defined custom scalar fields (not associated with PDE solvers)
        :return: None
        """
        import CompuCellSetup
        scalarFieldsDict = CompuCellSetup.fieldRegistry.getScalarFields()
        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectType == 'ScalarField' and sd.moduleType == 'Python':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                tmpFileName = sd.fileName
                sd.fileName = fullPath

                try:
                    sd.objectPtr = scalarFieldsDict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadScalarField(sd)
                sd.fileName = tmpFileName

    def loadScalarFieldsCellLevel(self):
        """
        Loads user-defined custom scalar fields (not associated with PDE solvers) that are defined on per-cell basis
        :return: None
        """

        import CompuCellSetup

        scalarFieldsDictCellLevel = CompuCellSetup.fieldRegistry.getScalarFieldsCellLevel()
        for resourceName, sd in self.__restartResourceDict.items():

            if sd.objectType == 'ScalarFieldCellLevel' and sd.moduleType == 'Python':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                tmpFileName = sd.fileName
                sd.fileName = fullPath

                try:
                    sd.objectPtr = scalarFieldsDictCellLevel[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadScalarFieldCellLevel(sd)
                sd.fileName = tmpFileName

    def loadVectorFields(self):
        """
        restores user-defined custom vector fields
        :return: None
        """

        import CompuCellSetup
        vectorFieldsDict = CompuCellSetup.fieldRegistry.getVectorFields()
        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectType == 'VectorField' and sd.moduleType == 'Python':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                tmpFileName = sd.fileName
                sd.fileName = fullPath

                try:
                    sd.objectPtr = vectorFieldsDict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadVectorField(sd)
                sd.fileName = tmpFileName

    def loadVectorFieldsCellLevel(self):
        """
        Loads user-defined custom vector fields that are defined on per-cell basis
        :return: None
        """
        import CompuCellSetup

        vectorFieldsCellLevelDict = CompuCellSetup.fieldRegistry.getVectorFieldsCellLevel()

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectType == 'VectorFieldCellLevel' and sd.moduleType == 'Python':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                tmpFileName = sd.fileName
                sd.fileName = fullPath

                try:
                    sd.objectPtr = vectorFieldsCellLevelDict[sd.objectName]

                except LookupError as e:
                    continue

                self.serializer.loadVectorFieldCellLevel(sd)
                sd.fileName = tmpFileName

    def loadCoreCellAttributes(self):
        """
        Loads core cell attributes such as lambdaVolume, targetVolume etc...
        :return: None
        """
        import pickle
        from .PySteppables import CellList

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'CoreCellAttributes' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)
                for cell in cellList:
                    cellId = pickle.load(pf)
                    clusterId = cell.clusterId
                    # print 'cellId=',cellId
                    cellCoreAttributes = pickle.load(pf)
                    # print 'cellCoreAttributes=',cellCoreAttributes
                    # cellLocal=inventory.getCellByIds(cellId,clusterId)   

                    if cell:
                        # print 'cell=',cell," cell.id=",cell.id    
                        self.setCellCoreAttributes(cell, cellCoreAttributes)

                pf.close()

    def unpickleDict(self, _fileName, _cellList):
        """
        Utility function that unpickles dictionary representing dictionary of attributes that user
        attaches to cells at the Python level
        :param _fileName: {str}
        :param _cellList: {container with all CC3D cells - equivalent of self.cellList in SteppableBasePy}
        :return:
        """
        import CompuCell
        import pickle
        import copy
        try:
            pf = open(_fileName, 'r')
        except IOError as e:
            return

        numberOfCells = pickle.load(pf)

        for cell in _cellList:
            cellId = pickle.load(pf)

            unpickledAttribDict = pickle.load(pf)

            dictAttrib = CompuCell.getPyAttrib(cell)

            # dictAttrib=copy.deepcopy(unpickledAttribDict)
            dictAttrib.update(
                unpickledAttribDict)  # adds all objects from unpickledAttribDict to dictAttrib -  note: deep copy will not work here

        pf.close()

    def unpickleList(self, _fileName, _cellList):
        """
        Utility function that unpickles list representing list of attributes that user
        attaches to cells at the Python level

        :param _fileName: {ste}
        :param _cellList: {container with all CC3D cells - equivalent of self.cellList in SteppableBasePy}
        :return:
        """

        import CompuCell
        import pickle

        try:
            pf = open(_fileName, 'r')
        except IOError as e:
            return

        numberOfCells = pickle.load(pf)

        for cell in _cellList:
            cellId = pickle.load(pf)
            unpickledAttribList = pickle.load(pf)
            listAttrib = CompuCell.getPyAttrib(cell)

            # appends all elements of unpickledAttribList to the end of listAttrib
            #  note: deep copy will not work here
            listAttrib.extend(unpickledAttribList)

        pf.close()

    def loadPythonAttributes(self):
        """
        Loads python attributes that user attached to cells (a list or dictionary)
        :return: None
        """

        import CompuCellSetup
        import pickle

        from .PySteppables import CellList

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'PythonAttributes' and sd.objectType == 'Pickle':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                # checking if cells have extra attribute
                import CompuCell
                for cell in cellList:
                    if not CompuCell.isPyAttribValid(cell):
                        return

                listFlag = True
                for cell in cellList:
                    attrib = CompuCell.getPyAttrib(cell)
                    if isinstance(attrib, list):
                        listFlag = True
                    else:
                        listFlag = False
                    break

                if listFlag:
                    self.unpickleList(fullPath, cellList)
                else:
                    self.unpickleDict(fullPath, cellList)

    def loadSBMLSolvers(self):
        """
        Loads SBML solvers
        :return: None
        """

        import CompuCellSetup
        import pickle
        from .PySteppables import CellList

        # loading and initializing freeFloating SBML Simulators
        #  SBML solvers associated with cells are loaded (but not fully initialized) in the loadPythonAttributes
        for resourceName, sd in self.__restartResourceDict.items():
            print('resourceName=', resourceName)
            print('sd=', sd)

            if sd.objectName == 'FreeFloatingSBMLSolvers' and sd.objectType == 'Pickle':
                print('RESTORING FreeFloatingSBMLSolvers ')

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                with open(fullPath, 'r') as pf:
                    CompuCellSetup.freeFloatingSBMLSimulator = pickle.load(pf)

                # initializing  freeFloating SBML Simulators       
                for modelName, sbmlSolver in CompuCellSetup.freeFloatingSBMLSimulator.items():
                    sbmlSolver.loadSBML(_externalPath=self.sim.getBasePath())

        # full initializing SBML solvers associated with cell
        #  we do that regardless whether we have freeFloatingSBMLSolver pickled file or not
        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)

        # checking if cells have extra attribute
        import CompuCell
        for cell in cellList:
            if not CompuCell.isPyAttribValid(cell):
                return
            else:
                attrib = CompuCell.getPyAttrib(cell)
                if isinstance(attrib, list):
                    return
                else:
                    break

        for cell in cellList:

            cellDict = CompuCell.getPyAttrib(cell)
            try:
                sbmlDict = cellDict['SBMLSolver']
                print('sbmlDict=', sbmlDict)
            except LookupError as e:
                continue

            for modelName, sbmlSolver in sbmlDict.items():
                # this call fully initializes SBML Solver by
                # loadSBML
                # ( relative path stored in sbmlSolver.path and root dir is passed using self.sim.getBasePath())
                sbmlSolver.loadSBML(_externalPath=self.sim.getBasePath())

    def loadBionetSolver(self):
        """
        Deprecated - loads bionet solver - SBMLSolver replaced this one
        :return: None
        """

        warnings.warn('BionetSolver is deprecated', PendingDeprecationWarning)

        import CompuCellSetup
        import pickle
        from .PySteppables import CellList

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'BionetSolver' and sd.objectType == 'Pickle':

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format

                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                    # first will load sbml files and
                import bionetAPI
                sbmlModelDict = pickle.load(pf)

                for modelName, modelDict in sbmlModelDict.items():
                    bionetAPI.loadSBMLModel(modelName, modelDict["ModelPath"], modelDict["ModelKey"],
                                            modelDict["ModelTimeStep"])

                # loading library names (model names)  associated with cell types
                cellTemplateLibraryDict = pickle.load(pf)

                # templateLibraryName in this case is the sdame as cell type name (except medium)
                for templateLibraryName, modelNames in cellTemplateLibraryDict.items():
                    for modelName in modelNames:
                        bionetAPI.addSBMLModelToTemplateLibrary(modelName, templateLibraryName)

                nonCellTemplateLibraryDict = pickle.load(pf)
                for nonCellLibName, modelInstanceDict in nonCellTemplateLibraryDict.items():
                    for modelName, varDict in modelInstanceDict.items():
                        bionetAPI.addSBMLModelToTemplateLibrary(modelName, nonCellLibName)

                bionetAPI.initializeBionetworks()

                # after bionetworks are initialized inthe bionetAPI we can assign variables to non cell models
                for nonCellLibName, modelInstanceDict in nonCellTemplateLibraryDict.items():
                    for modelName, varDict in modelInstanceDict.items():
                        for varName, varValue in varDict.items():
                            bionetAPI.setBionetworkValue(varName, varValue, nonCellLibName)

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                # checking if cells have extra attribute
                import CompuCell
                for cell in cellList:

                    dictAttrib = CompuCell.getPyAttrib(cell)
                    dictToPickle = {}

                    id = pickle.load(pf)  # cell id
                    cellSBMLModelData = pickle.load(pf)  # cell's sbml models

                    for modelName, modeVarDict in cellSBMLModelData.items():
                        for varName, varValue in modeVarDict.items():
                            bionetAPI.setBionetworkValue(varName, varValue, cell.id)

                pf.close()

    def loadAdhesionFlex(self):
        """
        restores AdhesionFlex Plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # AdhesionFlexPlugin

        adhesionFlexPlugin = None
        if self.sim.pluginManager.isLoaded("AdhesionFlex"):
            import CompuCell
            adhesionFlexPlugin = CompuCell.getAdhesionFlexPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'AdhesionFlex' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)
                # read medium adhesion molecule vector
                mediumAdhesionVector = pickle.load(pf)

                adhesionFlexPlugin.assignNewMediumAdhesionMoleculeDensityVector(mediumAdhesionVector)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    cellAdhesionVector = pickle.load(pf)
                    adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(cell, cellAdhesionVector)

                pf.close()
            adhesionFlexPlugin.overrideInitialization()

    def loadChemotaxis(self):
        """
        restores Chemotaxis
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # chemotaxisPlugin
        chemotaxisPlugin = None
        if self.sim.pluginManager.isLoaded("Chemotaxis"):
            import CompuCell
            chemotaxisPlugin = CompuCell.getChemotaxisPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'Chemotaxis' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    # loading number of chemotaxis data that cell has
                    chdNumber = pickle.load(pf)

                    for i in range(chdNumber):
                        # reading chemotaxis data 
                        chdDict = pickle.load(pf)
                        # creating chemotaxis data for cell
                        chd = chemotaxisPlugin.addChemotaxisData(cell, chdDict['fieldName'])
                        chd.setLambda(chdDict['lambda'])
                        chd.saturationCoef = chdDict['saturationCoef']
                        chd.setChemotaxisFormulaByName(chdDict['formulaName'])
                        chd.assignChemotactTowardsVectorTypes(chdDict['chemotactTowardsTypesVec'])

                pf.close()

    def loadLengthConstraint(self):
        """
        Restores LengthConstraint
        :return: None
        """
        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # LengthConstraintPlugin
        lengthConstraintPlugin = None
        if self.sim.pluginManager.isLoaded("LengthConstraint"):
            import CompuCell
            lengthConstraintPlugin = CompuCell.getLengthConstraintPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'LengthConstraint' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    lengthConstraintVec = pickle.load(pf)
                    # ([lcp.getLambdaLength(cell),lcp.getTargetLength(cell),lcp.getMinorTargetLength(cell)],pf)
                    lengthConstraintPlugin.setLengthConstraintData(cell, lengthConstraintVec[0], lengthConstraintVec[1],
                                                                   lengthConstraintVec[2])

                pf.close()

    def loadConnectivityGlobal(self):
        """
        Restores ConnectivityGlobal plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # ConnectivityLocalFlexPlugin
        connectivityGlobalPlugin = None
        if self.sim.pluginManager.isLoaded("ConnectivityGlobal"):
            import CompuCell
            connectivityGlobalPlugin = CompuCell.getConnectivityGlobalPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'ConnectivityGlobal' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    connectivityStrength = pickle.load(pf)
                    connectivityGlobalPlugin.setConnectivityStrength(cell, connectivityStrength)

                pf.close()

    def loadConnectivityLocalFlex(self):
        """
        Restores ConnectivityLocalFlex plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # ConnectivityLocalFlexPlugin
        connectivityLocalFlexPlugin = None
        if self.sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            import CompuCell
            connectivityLocalFlexPlugin = CompuCell.getConnectivityLocalFlexPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'ConnectivityLocalFlex' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    connectivityStrength = pickle.load(pf)
                    connectivityLocalFlexPlugin.setConnectivityStrength(cell, connectivityStrength)

                pf.close()

    def loadFocalPointPlasticity(self):
        """
        restores FocalPointPlasticity plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # FocalPointPlasticity
        focalPointPlasticityPlugin = None
        if self.sim.pluginManager.isLoaded("FocalPointPlasticity"):
            import CompuCell
            focalPointPlasticityPlugin = CompuCell.getFocalPointPlasticityPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'FocalPointPlasticity' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    cellId = cell.id
                    clusterId = cell.clusterId

                    # read number of fpp links in the cell (external)
                    linksNumber = pickle.load(pf)
                    for i in range(linksNumber):
                        fppDict = pickle.load(pf)  # loading external links
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        neighborIds = fppDict['neighborIds']  # cellId, cluster id

                        neighborCell = inventory.getCellByIds(neighborIds[0], neighborIds[1])
                        fpptd.neighborAddress = neighborCell
                        fpptd.lambdaDistance = fppDict['lambdaDistance']
                        fpptd.targetDistance = fppDict['targetDistance']
                        fpptd.maxDistance = fppDict['maxDistance']
                        fpptd.activationEnergy = fppDict['activationEnergy']
                        fpptd.maxNumberOfJunctions = fppDict['maxNumberOfJunctions']
                        fpptd.neighborOrder = fppDict['neighborOrder']

                        focalPointPlasticityPlugin.insertFPPData(cell, fpptd)

                    # read number of fpp links in the cell (internal)
                    internalLinksNumber = pickle.load(pf)
                    for i in range(internalLinksNumber):
                        fppDict = pickle.load(pf)  # loading external links
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        neighborIds = fppDict['neighborIds']  # cellId, cluster id
                        neighborCell = inventory.getCellByIds(neighborIds[0], neighborIds[1])
                        fpptd.neighborAddfess = neighborCell
                        fpptd.lambdaDistance = fppDict['lambdaDistance']
                        fpptd.targetDistance = fppDict['targetDistance']
                        fpptd.maxDistance = fppDict['maxDistance']
                        fpp.activationEnergy = fppDict['activationEnergy']
                        fpptd.maxNumberOfJunctions = fppDict['maxNumberOfJunctions']
                        fpptd.neighborOrder = fppDict['neighborOrder']
                        focalPointPlasticityPlugin.insertInternalFPPData(cell, fpptd)

                    # read number of fpp links in the cell (anchors)
                    anchorLinksNumber = pickle.load(pf)
                    for i in range(anchorLinksNumber):
                        fppDict = pickle.load(pf)  # loading external links
                        fpptd = CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        # neighborIds=fppDict['neighborIds'] # cellId, cluster id
                        # neighborCell=inventory.getCellByIds(neighborIds[0],neighborIds[1])
                        fpptd.neighborAddfess = 0
                        fpptd.lambdaDistance = fppDict['lambdaDistance']
                        fpptd.targetDistance = fppDict['targetDistance']
                        fpptd.maxDistance = fppDict['maxDistance']
                        fpptd.anchorId = fppDict['anchorId']
                        fpptd.anchorPoint[0] = fppDict['anchorPoint'][0]
                        fpptd.anchorPoint[1] = fppDict['anchorPoint'][1]
                        fpptd.anchorPoint[2] = fppDict['anchorPoint'][2]

                        focalPointPlasticityPlugin.insertAnchorFPPData(cell, fpptd)

                pf.close()

    def loadContactLocalProduct(self):
        """
        restores ContactLocalProduct plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # ContactLocalProductPlugin
        contactLocalProductPlugin = None
        if self.sim.pluginManager.isLoaded("ContactLocalProduct"):
            import CompuCell
            contactLocalProductPlugin = CompuCell.getContactLocalProductPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'ContactLocalProduct' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)
                    cadherinVector = pickle.load(pf)
                    contactLocalProductPlugin.setCadherinConcentrationVec(cell,
                                                                          CompuCell.contactproductdatacontainertype(
                                                                              cadherinVector))

                pf.close()

    def loadCellOrientation(self):
        """
        restores CellOriencation plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # CellOrientationPlugin
        cellOrientationPlugin = None
        if self.sim.pluginManager.isLoaded("CellOrientation"):
            import CompuCell
            cellOrientationPlugin = CompuCell.getCellOrientationPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'CellOrientation' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)
                    lambdaCellOrientation = pickle.load(pf)
                    cellOrientationPlugin.setLambdaCellOrientation(cell, lambdaCellOrientation)

                pf.close()

    def loadPolarizationVector(self):
        """
        restores polarizationVector plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # PolarizationVectorPlugin
        polarizationVectorPlugin = None
        if self.sim.pluginManager.isLoaded("PolarizationVector"):
            import CompuCell
            polarizationVectorPlugin = CompuCell.getPolarizationVectorPlugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'PolarizationVector' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)
                    polarizationVec = pickle.load(pf)
                    polarizationVectorPlugin.setPolarizationVector(cell, polarizationVec[0], polarizationVec[1],
                                                                   polarizationVec[2])

                pf.close()

    def loadPolarization23(self):
        """
        restores polarization23 plugin
        :return: None
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # polarization23Plugin
        polarization23Plugin = None
        if self.sim.pluginManager.isLoaded("Polarization23"):
            import CompuCell
            polarization23Plugin = CompuCell.getPolarization23Plugin()
        else:
            return

        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'Polarization23' and sd.objectType == 'Pickle':

                inventory = self.sim.getPotts().getCellInventory()
                cellList = CellList(inventory)

                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format
                try:
                    pf = open(fullPath, 'r')
                except IOError as e:
                    return

                numberOfCells = pickle.load(pf)

                for cell in cellList:
                    cellId = pickle.load(pf)

                    polVec = pickle.load(pf)  # [fX,fY,fZ]
                    polMarkers = pickle.load(pf)
                    lambdaPol = pickle.load(pf)

                    polarization23Plugin.setPolarizationVector(cell, CompuCell.Vector3(polVec[0], polVec[2], polVec[2]))
                    polarization23Plugin.setPolarizationMarkers(cell, polMarkers[0], polMarkers[1])
                    polarization23Plugin.setLambdaPolarization(cell, lambdaPol)

                pf.close()

    def outputSteeringPanel(self, _restartOutputPath, _rstXMLElem):
        """
        Outputs steering panel for python parameters
        :param restartOutputPath:{str} path to restart dir
        :param _rstXMLElem: {xml elem obj}
        :return: None
        """
        import SerializerDEPy

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'SteeringPanel'
        sd.moduleType = 'SteeringPanel'
        sd.objectName = 'SteeringPanel'
        sd.objectType = 'JSON'
        sd.fileName = os.path.join(_restartOutputPath, 'SteeringPanel' + '.json')

        import CompuCellSetup
        CompuCellSetup.serialize_steering_panel(sd.fileName)

        self.appendXMLStub(_rstXMLElem, sd)

    def loadSteeringPanel(self):
        """
        Deserializes steering panel
        :return:None
        """
        for resourceName, sd in self.__restartResourceDict.items():
            if sd.objectName == 'SteeringPanel' and sd.objectType.lower() == 'json':
                fullPath = os.path.join(self.__restartDirectory, sd.fileName)
                fullPath = os.path.abspath(fullPath)  # normalizing path format

                import CompuCellSetup
                CompuCellSetup.deserialize_steering_panel(fname=fullPath)

    def outputRestartFiles(self, _step=0, _onDemand=False):
        """
        main function that serializes simulation
        :param _step: {int} current MCS
        :param _onDemand: {False} flag representing whether serialization is ad-hoc or regularly scheduled one
        :return: None
        """

        if not _onDemand and self.__outputFrequency <= 0:
            return

        if not _onDemand and _step == 0:
            return

        if not _onDemand and _step % self.__outputFrequency:
            return

        # have to initialize serialized each time in case lattice gets resized in which case cellField Ptr
        # has to be updated and lattice dimension is usually different

        pg = CompuCellSetup.persistent_globals

        self.serializer.init(pg.simulator)

        rst_xml_elem = ElementCC3D("RestartFiles",
                                 {"Version": Version.getVersionAsString(), 'Build': Version.getSVNRevisionAsString()})
        rst_xml_elem.ElementCC3D("Step", {}, _step)
        print('outputRestartFiles')

        # cc3dSimOutputDir = CompuCellSetup.screenshotDirectoryName
        cc3dSimOutputDir = pg.output_directory

        print("cc3dSimOutputDir=", cc3dSimOutputDir)
        # print("CompuCellSetup.simulationPaths.simulationXMLFileName=", CompuCellSetup.simulationPaths.simulationXMLFileName)
        # print('CompuCellSetup.simulationFileName=', CompuCellSetup.simulationFileName)

        restart_output_path = self.setup_restart_output_directory(_step)

        # no output if restart_output_path is not specified
        if restart_output_path == '':
            return

        # ---------------------- OUTPUTTING RESTART FILES    --------------------
        # outputting cell field    
        self.output_cell_field(restart_output_path, rst_xml_elem)
        
        # outputting concentration fields (scalar fields) from PDE solvers    
        self.output_concentration_fields(restart_output_path, rst_xml_elem)
        
        # outputting extra scalar fields   - used in Python only
        self.output_scalar_fields(restart_output_path, rst_xml_elem)
        
        # outputting extra scalar fields cell level  - used in Python only
        self.output_scalar_fields_cell_level(restart_output_path, rst_xml_elem)
        
        # outputting extra vector fields  - used in Python only
        self.output_vector_fields(restart_output_path, rst_xml_elem)
        
        # outputting extra vector fields cell level  - used in Python only
        self.output_vector_fields_cell_level(restart_output_path, rst_xml_elem)
        
        # outputting core cell  attributes
        self.output_core_cell_attributes(restart_output_path, rst_xml_elem)
        
        # outputting cell Python attributes
        self.output_python_attributes(restart_output_path, rst_xml_elem)

        # outputting FreeFloating SBMLSolvers -
        # notice that SBML solvers assoaciated with a cell are pickled in the outputPythonAttributes function
        self.output_free_floating_sbml_solvers(restart_output_path, rst_xml_elem)
        
        # outputting plugins
        
        # outputting AdhesionFlexPlugin
        self.output_adhesion_flex_plugin(restart_output_path, rst_xml_elem)
        
        # outputting ChemotaxisPlugin
        self.output_chemotaxis_plugin(restart_output_path, rst_xml_elem)
        
        # outputting LengthConstraintPlugin
        self.output_length_constraint_plugin(restart_output_path, rst_xml_elem)
        
        # outputting ConnectivityGlobalPlugin
        self.output_connectivity_global_plugin(restart_output_path, rst_xml_elem)
        
        # outputting ConnectivityLocalFlexPlugin
        self.output_connectivity_local_flex_plugin(restart_output_path, rst_xml_elem)
        
        # outputting FocalPointPlacticityPlugin
        self.output_focal_point_placticity_plugin(restart_output_path, rst_xml_elem)
        
        # outputting ContactLocalProductPlugin
        self.output_contact_local_product_plugin(restart_output_path, rst_xml_elem)
        
        # outputting CellOrientationPlugin
        self.output_cell_orientation_plugin(restart_output_path, rst_xml_elem)
        
        # outputting PolarizationVectorPlugin
        self.output_polarization_vector_plugin(restart_output_path, rst_xml_elem)
        
        # outputting Polarization23Plugin
        self.output_polarization23_plugin(restart_output_path, rst_xml_elem)
        #
        # # outputting steering panel params
        # self.outputSteeringPanel(restart_output_path, rst_xml_elem)
        #
        # # ---------------------- END OF  OUTPUTTING RESTART FILES    --------------------
        #
        # # -------------writing xml description of the restart files
        # rst_xml_elem.CC3DXMLElement.saveXML(os.path.join(restart_output_path, 'restart.xml'))

        # --------------- depending on removePreviousFiles we will remove or keep previous restart files

        print('\n\n\n\n self.__allowMultipleRestartDirectories=', self.__allowMultipleRestartDirectories)

        if not self.__allowMultipleRestartDirectories:

            print('\n\n\n\n self.__completedRestartOutputPath=', self.__completedRestartOutputPath)

            if self.__completedRestartOutputPath != '':
                import shutil
                try:
                    shutil.rmtree(self.__completedRestartOutputPath)
                except:
                    # will ignore exceptions during directory removal -
                    # they might be due e.g. user accessing directory to be removed -
                    # in such a case it is best to ignore such requests
                    pass

        self.__completedRestartOutputPath = self.getRestartOutputRootPath(restart_output_path)

    def output_concentration_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes concentration fields (associated with PDE solvers)
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """

        sim = CompuCellSetup.persistent_globals.simulator
        conc_field_name_vec = sim.getConcentrationFieldNameVector()
        for fieldName in conc_field_name_vec:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'PDESolver'
            sd.moduleType = 'Steppable'

            sd.objectName = fieldName
            sd.objectType = 'ConcentrationField'
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            print('sd.fileName=', sd.fileName)
            sd.fileFormat = 'text'
            self.serializeDataList.append(sd)
            self.serializer.serializeConcentrationField(sd)
            self.appendXMLStub(rst_xml_elem, sd)
            print("Got concentration field: ", fieldName)

    def output_cell_field(self, restart_output_path, rst_xml_elem):
        """
        Serializes cell field
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        concFieldNameVec = sim.getConcentrationFieldNameVector()
        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Potts3D'
        sd.moduleType = 'Core'

        sd.objectName = 'CellField'
        sd.objectType = 'CellField'
        sd.fileName = os.path.join(restart_output_path, sd.objectName + '.dat')
        sd.fileFormat = 'text'
        self.serializeDataList.append(sd)
        self.serializer.serializeCellField(sd)
        self.appendXMLStub(rst_xml_elem, sd)

    def output_scalar_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers)
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry

        scalar_fields_dict = field_registry.getScalarFields()
        for fieldName in scalar_fields_dict:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'ScalarField'
            sd.objectPtr = scalar_fields_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeScalarField(sd)
            self.appendXMLStub(rst_xml_elem, sd)

    def output_scalar_fields_cell_level(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers) that are
        defined on the per-cell basis
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry
        scalar_fields_dict_cell_level = field_registry.getScalarFieldsCellLevel()
        for fieldName in scalar_fields_dict_cell_level:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'ScalarFieldCellLevel'
            sd.objectPtr = scalar_fields_dict_cell_level[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeScalarFieldCellLevel(sd)
            self.appendXMLStub(rst_xml_elem, sd)

    def output_vector_fields(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined vector fields
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """

        field_registry = CompuCellSetup.persistent_globals.field_registry
        vector_fields_dict = field_registry.getVectorFields()
        for fieldName in vector_fields_dict:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'VectorField'
            sd.objectPtr = vector_fields_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeVectorField(sd)
            self.appendXMLStub(rst_xml_elem, sd)

    def output_vector_fields_cell_level(self, restart_output_path, rst_xml_elem):
        """
        Serializes user defined vector fields that are defined on per-cell basis
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        field_registry = CompuCellSetup.persistent_globals.field_registry
        vector_fields_cell_level_dict = field_registry.getVectorFieldsCellLevel()
        for fieldName in vector_fields_cell_level_dict:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'Python'
            sd.moduleType = 'Python'
            sd.objectName = fieldName
            sd.objectType = 'VectorFieldCellLevel'
            sd.objectPtr = vector_fields_cell_level_dict[fieldName]
            sd.fileName = os.path.join(restart_output_path, fieldName + '.dat')
            self.serializer.serializeVectorFieldCellLevel(sd)
            self.appendXMLStub(rst_xml_elem, sd)

    def cellCoreAttributes(self, _cell):
        """
        produces a dictionary containing core CellG attributes
        :param _cell:{instance of CellG object} cc3d cell
        :return: {dict}
        """

        coreAttribDict = {}
        coreAttribDict['targetVolume'] = _cell.targetVolume
        coreAttribDict['lambdaVolume'] = _cell.lambdaVolume
        coreAttribDict['targetSurface'] = _cell.targetSurface
        coreAttribDict['lambdaSurface'] = _cell.lambdaSurface
        coreAttribDict['targetClusterSurface'] = _cell.targetClusterSurface
        coreAttribDict['lambdaClusterSurface'] = _cell.lambdaClusterSurface
        coreAttribDict['type'] = _cell.type
        coreAttribDict['xCOMPrev'] = _cell.xCOMPrev
        coreAttribDict['yCOMPrev'] = _cell.yCOMPrev
        coreAttribDict['zCOMPrev'] = _cell.zCOMPrev
        coreAttribDict['lambdaVecX'] = _cell.lambdaVecX
        coreAttribDict['lambdaVecY'] = _cell.lambdaVecY
        coreAttribDict['lambdaVecZ'] = _cell.lambdaVecZ
        coreAttribDict['flag'] = _cell.flag
        coreAttribDict['fluctAmpl'] = _cell.fluctAmpl

        return coreAttribDict

    def setCellCoreAttributes(self, _cell, _coreAttribDict):
        """
        initializes cell attributes
        :param _cell: {instance of CellG object} cc3d cell
        :param _coreAttribDict: {dict} dictionry of attributes
        :return:
        """

        for attribName, attribValue in _coreAttribDict.items():

            try:
                setattr(_cell, attribName, attribValue)

            except LookupError as e:
                continue
            except AttributeError as ea:
                continue

    def output_core_cell_attributes(self, restart_output_path, rst_xml_elem):
        """
        Serializes core clel attributes - the ones from CellG C++ object such as lambdaVolume, targetVolume, etc...
        :param restart_output_path:{str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return: None
        """
        sim = CompuCellSetup.persistent_globals.simulator
        inventory = sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Potts3D'
        sd.moduleType = 'Core'
        sd.objectName = 'CoreCellAttributes'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'CoreCellAttributes' + '.dat')
        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)
        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(self.cellCoreAttributes(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def pickleList(self, _fileName, _cellList):
        """
        Utility function for pickling CellList object
        :param _fileName: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """
        numberOfCells = len(_cellList)

        nullFile = open(os.devnull, 'w')
        try:
            pf = open(_fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in _cellList:
            # print 'cell.id=',cell.id
            listAttrib = CompuCell.getPyAttrib(cell)
            listToPickle = []
            # checking which list items are picklable
            for item in listAttrib:
                try:
                    pickle.dump(item, nullFile)
                    listToPickle.append(item)
                except TypeError as e:
                    print("PICKLNG LIST")
                    print(e)
                    pass

            pickle.dump(cell.id, pf)
            pickle.dump(listToPickle, pf)

        nullFile.close()
        pf.close()

    def pickleDictionary(self, _fileName, _cellList):
        """
        Utility function for pickling list of attributes attached to cells by user in the Python script
        :param _fileName: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """

        numberOfCells = len(_cellList)

        nullFile = open(os.devnull, 'wb')
        try:
            pf = open(_fileName, 'wb')
        except IOError as e:
            return

        # --------------------------
        # pt=CompuCell.Vector3(10,11,12)

        # pf1=open('PickleCC3D.dat','w')
        # cPickle.dump(pt,pf1)

        # pf1.close()

        # pf1=open('PickleCC3D.dat','r')

        # content=cPickle.load(pf1)
        # print 'content=',content
        # print 'type(content)=',type(content)
        # pf1.close()
        # --------------------------

        pickle.dump(numberOfCells, pf)

        for cell in _cellList:
            # print 'cell.id=',cell.id
            dictAttrib = CompuCell.getPyAttrib(cell)
            dictToPickle = {}
            # checking which list items are picklable
            for key in dictAttrib:
                try:
                    pickle.dump(dictAttrib[key], nullFile)
                    dictToPickle[key] = dictAttrib[key]

                except TypeError as e:
                    print("key=", key, " cannot be pickled")
                    print(e)
                    pass

            pickle.dump(cell.id, pf)
            pickle.dump(dictToPickle, pf)

        nullFile.close()
        pf.close()

    def output_free_floating_sbml_solvers(self, restart_output_path, rst_xml_elem):

        """
        Outputs free-floating SBML solvers
        :param  restart_output_path: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """

        free_floating_sbml_simulators = CompuCellSetup.persistent_globals.free_floating_sbml_simulators

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'FreeFloatingSBMLSolvers'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'FreeFloatingSBMLSolvers' + '.dat')
        # checking if freeFloatingSBMLSimulator is non-empty
        if free_floating_sbml_simulators:
            with open(sd.fileName, 'w') as pf:
                pickle.dump(free_floating_sbml_simulators, pf)
                self.appendXMLStub(rst_xml_elem, sd)

    def output_python_attributes(self, restart_output_path, rst_xml_elem):
        """
        outputs python attributes that were attached to a cell by the user in the Python script
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """
        sim = CompuCellSetup.persistent_globals.simulator
        # notice that this function also outputs SBMLSolver objects
        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)

        # checking if cells have extra attribute

        for cell in cell_list:
            if not CompuCell.isPyAttribValid(cell):
                return

        list_flag = True
        for cell in cell_list:
            attrib = CompuCell.getPyAttrib(cell)
            if isinstance(attrib, list):
                list_flag = True
            else:
                list_flag = False
            break

        print('list_flag=', list_flag)

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'PythonAttributes'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'PythonAttributes' + '.dat')
        # cPickle.dump(numberOfCells,pf)

        if list_flag:
            self.pickleList(sd.fileName, cell_list)
        else:
            self.pickleDictionary(sd.fileName, cell_list)

        self.appendXMLStub(rst_xml_elem, sd)

    def output_adhesion_flex_plugin(self, restart_output_path, rst_xml_elem):
        """
        serializes AdhesionFlex Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("AdhesionFlex"):
            return

        adhesion_flex_plugin = CompuCell.getAdhesionFlexPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'AdhesionFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'AdhesionFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'AdhesionFlex' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)
        # wtiting medium adhesion vector

        medium_adhesion_vector = adhesion_flex_plugin.getMediumAdhesionMoleculeDensityVector()
        pickle.dump(medium_adhesion_vector, pf)
        for cell in cell_list:
            pickle.dump(cell.id, pf)
            cell_adhesion_vector = adhesion_flex_plugin.getAdhesionMoleculeDensityVector(cell)
            pickle.dump(cell_adhesion_vector, pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_chemotaxis_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes Chemotaxis Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("Chemotaxis"):
            return
        chemotaxis_plugin = CompuCell.getChemotaxisPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Chemotaxis'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Chemotaxis'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'Chemotaxis' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)
        for cell in cell_list:
            pickle.dump(cell.id, pf)

            field_names = chemotaxis_plugin.getFieldNamesWithChemotaxisData(cell)
            # outputting numbed of chemotaxis data that cell has
            pickle.dump(len(field_names), pf)

            for fieldName in field_names:
                chd = chemotaxis_plugin.getChemotaxisData(cell, fieldName)
                chd_dict = {}
                chd_dict['fieldName'] = fieldName
                chd_dict['lambda'] = chd.getLambda()
                chd_dict['saturationCoef'] = chd.saturationCoef
                chd_dict['formulaName'] = chd.formulaName
                chemotactTowardsVec = chd.getChemotactTowardsVectorTypes()
                print('chemotactTowardsVec=', chemotactTowardsVec)
                chd_dict['chemotactTowardsTypesVec'] = chd.getChemotactTowardsVectorTypes()

                pickle.dump(chd_dict, pf)
            print('field_names=', field_names)
            # cPickle.dump(cellAdhesionVector,pf)        

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_length_constraint_plugin(self, restart_output_path, rst_xml_elem):
        """
        serializes LengthConstraint Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("LengthConstraint"):
            return
        length_constraint_plugin = CompuCell.getLengthConstraintPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'LengthConstraint'
        sd.moduleType = 'Plugin'
        sd.objectName = 'LengthConstraint'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'LengthConstraint' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        lcp = length_constraint_plugin

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump([lcp.getLambdaLength(cell), lcp.getTargetLength(cell), lcp.getMinorTargetLength(cell)], pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_connectivity_global_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ConnectivityGlobal Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityGlobal"):
            return

        connectivity_global_plugin = CompuCell.getConnectivityGlobalPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityGlobal'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityGlobal'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ConnectivityGlobal' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivity_global_plugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_connectivity_local_flex_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ConnectivityLocalFlex Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """
        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            return

        connectivity_local_flex_plugin = CompuCell.getConnectivityLocalFlexPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityLocalFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityLocalFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ConnectivityLocalFlex' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivity_local_flex_plugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_focal_point_placticity_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes FocalPointPlacticity Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("FocalPointPlasticity"):
            return

        focal_point_plasticity_plugin = CompuCell.getFocalPointPlasticityPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'FocalPointPlasticity'
        sd.moduleType = 'Plugin'
        sd.objectName = 'FocalPointPlasticity'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'FocalPointPlasticity' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:

            pickle.dump(cell.id, pf)
            fpp_vec = focal_point_plasticity_plugin.getFPPDataVec(cell)
            internal_fpp_vec = focal_point_plasticity_plugin.getInternalFPPDataVec(cell)
            anchor_fpp_vec = focal_point_plasticity_plugin.getAnchorFPPDataVec(cell)

            # dumping 'external' fpp links
            pickle.dump(len(fpp_vec), pf)
            for fpp_data in fpp_vec:
                fpp_data_dict = {}
                if fpp_data.neighborAddress:
                    fpp_data_dict['neighborIds'] = [fpp_data.neighborAddress.id, fpp_data.neighborAddress.clusterId]
                else:
                    fpp_data_dict['neighborIds'] = [0, 0]
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['activationEnergy'] = fpp_data.activationEnergy
                fpp_data_dict['maxNumberOfJunctions'] = fpp_data.maxNumberOfJunctions
                fpp_data_dict['neighborOrder'] = fpp_data.neighborOrder
                pickle.dump(fpp_data_dict, pf)

            # dumping 'internal' fpp links
            pickle.dump(len(internal_fpp_vec), pf)
            for fpp_data in internal_fpp_vec:
                fpp_data_dict = {}
                if fpp_data.neighborAddress:
                    fpp_data_dict['neighborIds'] = [fpp_data.neighborAddress.id, fpp_data.neighborAddress.clusterId]
                else:
                    fpp_data_dict['neighborIds'] = [0, 0]
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['activationEnergy'] = fpp_data.activationEnergy
                fpp_data_dict['maxNumberOfJunctions'] = fpp_data.maxNumberOfJunctions
                fpp_data_dict['neighborOrder'] = fpp_data.neighborOrder
                pickle.dump(fpp_data_dict, pf)

            # dumping anchor fpp links
            pickle.dump(len(anchor_fpp_vec), pf)
            for fpp_data in anchor_fpp_vec:
                fpp_data_dict = {}
                fpp_data_dict['lambdaDistance'] = fpp_data.lambdaDistance
                fpp_data_dict['targetDistance'] = fpp_data.targetDistance
                fpp_data_dict['maxDistance'] = fpp_data.maxDistance
                fpp_data_dict['anchorId'] = fpp_data.anchorId
                fpp_data_dict['anchorPoint'] = [fpp_data.anchorPoint[0], fpp_data.anchorPoint[1],
                                                fpp_data.anchorPoint[2]]
                pickle.dump(fpp_data_dict, pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_contact_local_product_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes ContactLocalProduct Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("ContactLocalProduct"):
            return

        contact_local_product_plugin = CompuCell.getContactLocalProductPlugin()
        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ContactLocalProduct'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ContactLocalProduct'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'ContactLocalProduct' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(contact_local_product_plugin.getCadherinConcentrationVec(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_cell_orientation_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes CellOrientation Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("CellOrientation"):
            return

        cell_orientation_plugin = CompuCell.getCellOrientationPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'CellOrientation'
        sd.moduleType = 'Plugin'
        sd.objectName = 'CellOrientation'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'CellOrientation' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(cell_orientation_plugin.getLambdaCellOrientation(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_polarization_vector_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes PolarizationVector Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
        if not sim.pluginManager.isLoaded("PolarizationVector"):
            return

        polarization_vector_plugin = CompuCell.getPolarizationVectorPlugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'PolarizationVector'
        sd.moduleType = 'Plugin'
        sd.objectName = 'PolarizationVector'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'PolarizationVector' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pickle.dump(polarization_vector_plugin.getPolarizationVector(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

    def output_polarization23_plugin(self, restart_output_path, rst_xml_elem):

        """
        serializes Polarization23 Plugin
        :param restart_output_path: {str}
        :param rst_xml_elem: {instance of CC3DXMLElement}
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator

        if not sim.pluginManager.isLoaded("Polarization23"):
            return

        polarization23_plugin = CompuCell.getPolarization23Plugin()

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Polarization23'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Polarization23'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(restart_output_path, 'Polarization23' + '.dat')

        inventory = sim.getPotts().getCellInventory()
        cell_list = CellList(inventory)
        number_of_cells = len(cell_list)

        try:
            pf = open(sd.fileName, 'wb')
        except IOError:
            return

        pickle.dump(number_of_cells, pf)

        for cell in cell_list:
            pickle.dump(cell.id, pf)
            pol_vec = polarization23_plugin.getPolarizationVector(cell)
            pickle.dump([pol_vec.fX, pol_vec.fY, pol_vec.fZ], pf)
            pickle.dump(polarization23_plugin.getPolarizationMarkers(cell), pf)
            pickle.dump(polarization23_plugin.getLambdaPolarization(cell), pf)

        pf.close()
        self.appendXMLStub(rst_xml_elem, sd)

