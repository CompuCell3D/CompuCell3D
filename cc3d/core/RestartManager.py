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
        self.sim = _sim

        self.serializer = SerializerDEPy.SerializerDE()
        self.serializer.init(self.sim)
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

        #todo - fix
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
        output_dir_root  = pg.output_directory
        if not self.__step_number_of_digits:
            self.__step_number_of_digits = len(str(pg.simulator.getNumSteps()))

        restart_output_dir = Path(output_dir_root).joinpath('restart_'+str(_step).zfill(self.__step_number_of_digits))
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

                print('\n\n\n\n cc3dSimulationDataHandlerLocal.cc3dSimulationData=', cc3dSimulationDataHandlerLocal.cc3dSimulationData)

                # update simulation size in the XML  in case it has changed during the simulation 
                if cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript != '':
                    print('cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript=', cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript)
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
            print("cc3dSimulationDataHandler.cc3dSimulationData.serializerResource=", cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory)
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


        rstXMLElem = ElementCC3D("RestartFiles",
                                 {"Version": Version.getVersionAsString(), 'Build': Version.getSVNRevisionAsString()})
        rstXMLElem.ElementCC3D("Step", {}, _step)
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
        self.outputCellField(restart_output_path, rstXMLElem)
        # outputting concentration fields (scalar fields) from PDE solvers    
        self.outputConcentrationFields(restart_output_path, rstXMLElem)
        # outputting extra scalar fields   - used in Python only
        self.outputScalarFields(restart_output_path, rstXMLElem)
        # outputting extra scalar fields cell level  - used in Python only
        self.outputScalarFieldsCellLevel(restart_output_path, rstXMLElem)
        # outputting extra vector fields  - used in Python only
        self.outputVectorFields(restart_output_path, rstXMLElem)
        # outputting extra vector fields cell level  - used in Python only
        self.outputVectorFieldsCellLevel(restart_output_path, rstXMLElem)
        # outputting core cell  attributes
        self.outputCoreCellAttributes(restart_output_path, rstXMLElem)
        # outputting cell Python attributes
        self.outputPythonAttributes(restart_output_path, rstXMLElem)
        # return
        # # outputting FreeFloating SBMLSolvers -
        # # notice that SBML solvers assoaciated with a cell are pickled in the outputPythonAttributes function
        # self.outputFreeFloatingSBMLSolvers(restart_output_path, rstXMLElem)
        # # outputting plugins
        # # outputting AdhesionFlexPlugin
        # self.outputAdhesionFlexPlugin(restart_output_path, rstXMLElem)
        # # outputting ChemotaxisPlugin
        # self.outputChemotaxisPlugin(restart_output_path, rstXMLElem)
        # # outputting LengthConstraintPlugin
        # self.outputLengthConstraintPlugin(restart_output_path, rstXMLElem)
        # # outputting ConnectivityGlobalPlugin
        # self.outputConnectivityGlobalPlugin(restart_output_path, rstXMLElem)
        # # outputting ConnectivityLocalFlexPlugin
        # self.outputConnectivityLocalFlexPlugin(restart_output_path, rstXMLElem)
        # # outputting FocalPointPlacticityPlugin
        # self.outputFocalPointPlacticityPlugin(restart_output_path, rstXMLElem)
        # # outputting ContactLocalProductPlugin
        # self.outputContactLocalProductPlugin(restart_output_path, rstXMLElem)
        # # outputting CellOrientationPlugin
        # self.outputCellOrientationPlugin(restart_output_path, rstXMLElem)
        # # outputting PolarizationVectorPlugin
        # self.outputPolarizationVectorPlugin(restart_output_path, rstXMLElem)
        # # outputting Polarization23Plugin
        # self.outputPolarization23Plugin(restart_output_path, rstXMLElem)
        #
        # # outputting steering panel params
        # self.outputSteeringPanel(restart_output_path, rstXMLElem)
        #
        # # ---------------------- END OF  OUTPUTTING RESTART FILES    --------------------
        #
        # # -------------writing xml description of the restart files
        # rstXMLElem.CC3DXMLElement.saveXML(os.path.join(restart_output_path, 'restart.xml'))

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

    def outputConcentrationFields(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes concentration fields (associated with PDE solvers)
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return: None
        """

        concFieldNameVec = self.sim.getConcentrationFieldNameVector()
        for fieldName in concFieldNameVec:
            sd = SerializerDEPy.SerializeData()
            sd.moduleName = 'PDESolver'
            sd.moduleType = 'Steppable'

            sd.objectName = fieldName
            sd.objectType = 'ConcentrationField'
            sd.fileName = os.path.join(_restartOutputPath, fieldName + '.dat')
            print('sd.fileName=', sd.fileName)
            sd.fileFormat = 'text'
            self.serializeDataList.append(sd)
            self.serializer.serializeConcentrationField(sd)
            self.appendXMLStub(_rstXMLElem, sd)
            print("Got concentration field: ", fieldName)

    def outputCellField(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes cell field
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return: None
        """

        concFieldNameVec = self.sim.getConcentrationFieldNameVector()
        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Potts3D'
        sd.moduleType = 'Core'

        sd.objectName = 'CellField'
        sd.objectType = 'CellField'
        sd.fileName = os.path.join(_restartOutputPath, sd.objectName + '.dat')
        sd.fileFormat = 'text'
        self.serializeDataList.append(sd)
        self.serializer.serializeCellField(sd)
        self.appendXMLStub(_rstXMLElem, sd)

    def outputScalarFields(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers)
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
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
            sd.fileName = os.path.join(_restartOutputPath, fieldName + '.dat')
            self.serializer.serializeScalarField(sd)
            self.appendXMLStub(_rstXMLElem, sd)

    def outputScalarFieldsCellLevel(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes user defined scalar fields (not associated with PDE solvers) that are
        defined on the per-cell basis
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
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
            sd.fileName = os.path.join(_restartOutputPath, fieldName + '.dat')
            self.serializer.serializeScalarFieldCellLevel(sd)
            self.appendXMLStub(_rstXMLElem, sd)

    def outputVectorFields(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes user defined vector fields
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
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
            sd.fileName = os.path.join(_restartOutputPath, fieldName + '.dat')
            self.serializer.serializeVectorField(sd)
            self.appendXMLStub(_rstXMLElem, sd)

    def outputVectorFieldsCellLevel(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes user defined vector fields that are defined on per-cell basis
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
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
            sd.fileName = os.path.join(_restartOutputPath, fieldName + '.dat')
            self.serializer.serializeVectorFieldCellLevel(sd)
            self.appendXMLStub(_rstXMLElem, sd)

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

    def outputCoreCellAttributes(self, _restartOutputPath, _rstXMLElem):
        """
        Serializes core clel attributes - the ones from CellG C++ object such as lambdaVolume, targetVolume, etc...
        :param _restartOutputPath:{str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
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
        sd.fileName = os.path.join(_restartOutputPath, 'CoreCellAttributes' + '.dat')
        try:
            pf = open(sd.fileName, 'wb')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)
        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(self.cellCoreAttributes(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def pickleList(self, _fileName, _cellList):
        """
        Utility function for pickling CellList object
        :param _fileName: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """
        import CompuCell
        import pickle

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

    def outputFreeFloatingSBMLSolvers(self, _restartOutputPath, _rstXMLElem):

        """
        Outputs free-floating SBML solvers
        :param  _restartOutputPath: {str}
        :param _cellList: {instance of CellList} - a container representing all CC3D simulations
        :return: None
        """

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'FreeFloatingSBMLSolvers'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'FreeFloatingSBMLSolvers' + '.dat')
        if CompuCellSetup.freeFloatingSBMLSimulator:  # checking if freeFloatingSBMLSimulator is non-empty
            with open(sd.fileName, 'w') as pf:
                pickle.dump(CompuCellSetup.freeFloatingSBMLSimulator, pf)
                self.appendXMLStub(_rstXMLElem, sd)

    def outputPythonAttributes(self, _restartOutputPath, _rstXMLElem):
        """
        outputs python attributes that were attached to a cell by the user in the Python script
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
        """

        # notice that this function also outputs SBMLSolver objects
        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)

        # checking if cells have extra attribute

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

        print('listFlag=', listFlag)

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Python'
        sd.moduleType = 'Python'
        sd.objectName = 'PythonAttributes'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'PythonAttributes' + '.dat')
        # cPickle.dump(numberOfCells,pf)

        if listFlag:
            self.pickleList(sd.fileName, cellList)
        else:
            self.pickleDictionary(sd.fileName, cellList)

        self.appendXMLStub(_rstXMLElem, sd)

    def outputAdhesionFlexPlugin(self, _restartOutputPath, _rstXMLElem):
        """
        serializes AdhesionFlex Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'AdhesionFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'AdhesionFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'AdhesionFlex' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)
        # wtiting medium adhesion vector

        mediumAdhesionVector = adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
        pickle.dump(mediumAdhesionVector, pf)
        for cell in cellList:
            pickle.dump(cell.id, pf)
            cellAdhesionVector = adhesionFlexPlugin.getAdhesionMoleculeDensityVector(cell)
            pickle.dump(cellAdhesionVector, pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputChemotaxisPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes Chemotaxis Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
        """

        import SerializerDEPy
        import CompuCellSetup
        import pickle
        from .PySteppables import CellList
        import CompuCell

        # ChemotaxisPlugin
        chemotaxisPlugin = None
        if self.sim.pluginManager.isLoaded("Chemotaxis"):
            import CompuCell
            chemotaxisPlugin = CompuCell.getChemotaxisPlugin()
        else:
            return

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Chemotaxis'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Chemotaxis'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'Chemotaxis' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)
        for cell in cellList:
            pickle.dump(cell.id, pf)

            fieldNames = chemotaxisPlugin.getFieldNamesWithChemotaxisData(cell)
            # outputting numbed of chemotaxis data that cell has
            pickle.dump(len(fieldNames), pf)

            for fieldName in fieldNames:
                chd = chemotaxisPlugin.getChemotaxisData(cell, fieldName)
                chdDict = {}
                chdDict['fieldName'] = fieldName
                chdDict['lambda'] = chd.getLambda()
                chdDict['saturationCoef'] = chd.saturationCoef
                chdDict['formulaName'] = chd.formulaName
                chemotactTowardsVec = chd.getChemotactTowardsVectorTypes()
                print('chemotactTowardsVec=', chemotactTowardsVec)
                chdDict['chemotactTowardsTypesVec'] = chd.getChemotactTowardsVectorTypes()

                pickle.dump(chdDict, pf)
            print('fieldNames=', fieldNames)
            # cPickle.dump(cellAdhesionVector,pf)        

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputLengthConstraintPlugin(self, _restartOutputPath, _rstXMLElem):
        """
        serializes LengthConstraint Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'LengthConstraint'
        sd.moduleType = 'Plugin'
        sd.objectName = 'LengthConstraint'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'LengthConstraint' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        lcp = lengthConstraintPlugin

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump([lcp.getLambdaLength(cell), lcp.getTargetLength(cell), lcp.getMinorTargetLength(cell)], pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputConnectivityGlobalPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes ConnectivityGlobal Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityGlobal'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityGlobal'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'ConnectivityGlobal' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivityGlobalPlugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputConnectivityLocalFlexPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes ConnectivityLocalFlex Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ConnectivityLocalFlex'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ConnectivityLocalFlex'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'ConnectivityLocalFlex' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(connectivityLocalFlexPlugin.getConnectivityStrength(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputFocalPointPlacticityPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes FocalPointPlacticity Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'FocalPointPlasticity'
        sd.moduleType = 'Plugin'
        sd.objectName = 'FocalPointPlasticity'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'FocalPointPlasticity' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:

            pickle.dump(cell.id, pf)
            fppVec = focalPointPlasticityPlugin.getFPPDataVec(cell)
            internalFPPVec = focalPointPlasticityPlugin.getInternalFPPDataVec(cell)
            anchorFPPVec = focalPointPlasticityPlugin.getAnchorFPPDataVec(cell)

            # dumping 'external' fpp links
            pickle.dump(len(fppVec), pf)
            for fppData in fppVec:
                fppDataDict = {}
                if fppData.neighborAddress:
                    fppDataDict['neighborIds'] = [fppData.neighborAddress.id, fppData.neighborAddress.clusterId]
                else:
                    fppDataDict['neighborIds'] = [0, 0]
                fppDataDict['lambdaDistance'] = fppData.lambdaDistance
                fppDataDict['targetDistance'] = fppData.targetDistance
                fppDataDict['maxDistance'] = fppData.maxDistance
                fppDataDict['activationEnergy'] = fppData.activationEnergy
                fppDataDict['maxNumberOfJunctions'] = fppData.maxNumberOfJunctions
                fppDataDict['neighborOrder'] = fppData.neighborOrder
                pickle.dump(fppDataDict, pf)

            # dumping 'internal' fpp links
            pickle.dump(len(internalFPPVec), pf)
            for fppData in internalFPPVec:
                fppDataDict = {}
                if fppData.neighborAddress:
                    fppDataDict['neighborIds'] = [fppData.neighborAddress.id, fppData.neighborAddress.clusterId]
                else:
                    fppDataDict['neighborIds'] = [0, 0]
                fppDataDict['lambdaDistance'] = fppData.lambdaDistance
                fppDataDict['targetDistance'] = fppData.targetDistance
                fppDataDict['maxDistance'] = fppData.maxDistance
                fppDataDict['activationEnergy'] = fppData.activationEnergy
                fppDataDict['maxNumberOfJunctions'] = fppData.maxNumberOfJunctions
                fppDataDict['neighborOrder'] = fppData.neighborOrder
                pickle.dump(fppDataDict, pf)

            # dumping anchor fpp links
            pickle.dump(len(anchorFPPVec), pf)
            for fppData in anchorFPPVec:
                fppDataDict = {}
                fppDataDict['lambdaDistance'] = fppData.lambdaDistance
                fppDataDict['targetDistance'] = fppData.targetDistance
                fppDataDict['maxDistance'] = fppData.maxDistance
                fppDataDict['anchorId'] = fppData.anchorId
                fppDataDict['anchorPoint'] = [fppData.anchorPoint[0], fppData.anchorPoint[1], fppData.anchorPoint[2]]
                pickle.dump(fppDataDict, pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputContactLocalProductPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes ContactLocalProduct Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'ContactLocalProduct'
        sd.moduleType = 'Plugin'
        sd.objectName = 'ContactLocalProduct'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'ContactLocalProduct' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(contactLocalProductPlugin.getCadherinConcentrationVec(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputCellOrientationPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes CellOrientation Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'CellOrientation'
        sd.moduleType = 'Plugin'
        sd.objectName = 'CellOrientation'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'CellOrientation' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(cellOrientationPlugin.getLambdaCellOrientation(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputPolarizationVectorPlugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes PolarizationVector Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'PolarizationVector'
        sd.moduleType = 'Plugin'
        sd.objectName = 'PolarizationVector'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'PolarizationVector' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            pickle.dump(polarizationVectorPlugin.getPolarizationVector(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    def outputPolarization23Plugin(self, _restartOutputPath, _rstXMLElem):

        """
        serializes Polarization23 Plugin
        :param _restartOutputPath: {str}
        :param _rstXMLElem: {instance of CC3DXMLElement}
        :return:
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

        sd = SerializerDEPy.SerializeData()
        sd.moduleName = 'Polarization23'
        sd.moduleType = 'Plugin'
        sd.objectName = 'Polarization23'
        sd.objectType = 'Pickle'
        sd.fileName = os.path.join(_restartOutputPath, 'Polarization23' + '.dat')

        inventory = self.sim.getPotts().getCellInventory()
        cellList = CellList(inventory)
        numberOfCells = len(cellList)

        try:
            pf = open(sd.fileName, 'w')
        except IOError as e:
            return

        pickle.dump(numberOfCells, pf)

        for cell in cellList:
            pickle.dump(cell.id, pf)
            polVec = polarization23Plugin.getPolarizationVector(cell)
            pickle.dump([polVec.fX, polVec.fY, polVec.fZ], pf)
            pickle.dump(polarization23Plugin.getPolarizationMarkers(cell), pf)
            pickle.dump(polarization23Plugin.getLambdaPolarization(cell), pf)

        pf.close()
        self.appendXMLStub(_rstXMLElem, sd)

    # def outputPolarization23Plugin(self, _restartOutputPath, _rstXMLElem):
    #     """
    #     Serializes Polarization23PLugin
    #     :param _restartOutputPath: {str}
    #     :param _rstXMLElem: {instance of CC3DXMLElement}
    #     :return:
    #     """
    #
    #
    #     import SerializerDEPy
    #     import CompuCellSetup
    #     import cPickle
    #     from PySteppables import CellList
    #     import CompuCell
    #
    #     # polarization23Plugin
    #     polarization23Plugin = None
    #     if self.sim.pluginManager.isLoaded("Polarization23"):
    #         import CompuCell
    #         polarization23Plugin = CompuCell.getPolarization23Plugin()
    #     else:
    #         return
    #
    #     sd = SerializerDEPy.SerializeData()
    #     sd.moduleName = 'Polarization23'
    #     sd.moduleType = 'Plugin'
    #     sd.objectName = 'Polarization23'
    #     sd.objectType = 'Pickle'
    #     sd.fileName = os.path.join(_restartOutputPath, 'Polarization23' + '.dat')
    #
    #     inventory = self.sim.getPotts().getCellInventory()
    #     cellList = CellList(inventory)
    #     numberOfCells = len(cellList)
    #
    #     try:
    #         pf = open(sd.fileName, 'w')
    #     except IOError, e:
    #         return
    #
    #     cPickle.dump(numberOfCells, pf)
    #
    #     for cell in cellList:
    #         cPickle.dump(cell.id, pf)
    #         polVec = polarization23Plugin.getPolarizationVector(cell)
    #         cPickle.dump([polVec.fX, polVec.fY, polVec.fZ], pf)
    #         cPickle.dump(polarization23Plugin.getPolarizationMarkers(), pf)
    #         cPickle.dump(polarization23Plugin.getLambdaPolarization(), pf)
    #
    #     pf.close()
    #     self.appendXMLStub(_rstXMLElem, sd)
