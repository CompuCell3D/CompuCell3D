# -*- coding: utf-8 -*-
import os,sys


# MODULE_NAMES = ("AdhesionFlex", "ConField", "ScalarField", "ScalarFieldCellLevel" , "VectorField","VectorFieldCellLevel")
import CompuCell
def _pickleVector3(_vec):
    
    
    return CompuCell.Vector3 , (_vec.fX,_vec.fY,_vec.fZ)

    
    
import copy_reg
copy_reg.pickle(CompuCell.Vector3,_pickleVector3)    


class RestartManager:

    def __init__(self,_sim=None):
        import SerializerDEPy
        import CompuCellSetup
        self.sim=_sim
        self.serializer=SerializerDEPy.SerializerDE()
        self.serializer.init(self.sim)
        self.cc3dSimOutputDir=''
        self.serializeDataList=[]        
        self.__stepNumberOfDigits=0 # field size for formatting step number output
        self.__completedRestartOutputPath=''
        self.__allowMultipleRestartDirectories=True
        self.__outputFrequency=0
        self.__baseSimulationFilesCopied=False        
        
        # variables used during restarting
        self.__restartDirectory=''
        self.__restartFile=''        
        self.__restartVersion=0
        self.__restartBuild=0
        self.__restartStep=0
        self.__restartResourceDict={}
        
        self.cc3dSimulationDataHandler=None
                
        # self.extractRestartManagerConfiguration()
    def getRestartStep(self) :
        return self.__restartStep
        
    def prepareRestarter(self):
        import re
        import CompuCellSetup   
        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            import CC3DSimulationDataHandler        
            cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
            
            # checking is serializer resource exists
            if cc3dSimulationDataHandler.cc3dSimulationData.serializerResource:
                self.__allowMultipleRestartDirectories=cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.allowMultipleRestartDirectories
                self.__outputFrequency=cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.outputFrequency
        
    # def extractRestartManagerConfiguration(self):
        # import CompuCellSetup
        # metadataElem=CompuCellSetup.cc3dXML2ObjConverter.root.getFirstElement("Metadata")
        # if metadataElem:
            # serializeElem=metadataElem.getFirstElement("SerializeSimulation")
            
            # if serializeElem:
                # if serializeElem.findAttribute("OutputFrequency"):                        
                    # self.__outputFrequency=serializeElem.getAttributeAsInt("OutputFrequency")
                    
                # if serializeElem.findAttribute("AllowMultipleRestartDirectories"):                        
                    # self.__allowMultipleRestartDirectories=serializeElem.getAttributeAsBool("AllowMultipleRestartDirectories")
                    
    def restartEnabled(self):
        import CompuCellSetup
        import re
        # f=open('simcheck.dat','w')
        # print >>f,"REASTART MANAGER CompuCellSetup.simulationFileName=",CompuCellSetup.simulationFileName
        # f.close()
        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            print "EXTRACTING restartEnabled"
            import CC3DSimulationDataHandler        
            cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
            # print "cc3dSimulationDataHandler.cc3dSimulationData.serializerResource=",cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory
            return cc3dSimulationDataHandler.cc3dSimulationData.restartEnabled()
        return False    
    
    def appendXMLStub(selt,_rootElem,_sd):
        baseFileName=os.path.basename(_sd.fileName)
        attributeDict={"ModuleName":_sd.moduleName,"ModuleType":_sd.moduleType,"ObjectName":_sd.objectName,"ObjectType":_sd.objectType,"FileName":baseFileName,'FileFormat':_sd.fileFormat}
        _rootElem.ElementCC3D('ObjectData',attributeDict)
    
    
    def getRestartOutputRootPath(self,_restartOutputPath): 
        ''''
            returns path to the  is  output root directory e.g. <outputFolder>/restart_200
            '''
        restartOutputRootPath=os.path.dirname(_restartOutputPath)
        restartOutputRootPath=os.path.abspath(restartOutputRootPath) #normalizing path
        return restartOutputRootPath
        
    def setupRestartOutputDirectory(self,_step=0):
        import CompuCellSetup
        import string
        print 'CompuCellSetup.screenshotDirectoryName=',CompuCellSetup.screenshotDirectoryName
        self.cc3dSimOutputDir=CompuCellSetup.screenshotDirectoryName
        
        if not self.__stepNumberOfDigits:
            self.__stepNumberOfDigits=len(str(self.sim.getNumSteps()))
        
        restartOutputPath=''
        simFilesOutputPath=''
        if self.cc3dSimOutputDir=='':
            if str(CompuCellSetup.simulationFileName)!='':        
                (self.cc3dSimOutputDir, baseScreenshotName) = CompuCellSetup.makeSimDir(str(CompuCellSetup.simulationFileName))
                CompuCellSetup.screenshotDirectoryName=self.cc3dSimOutputDir                
                
                
                
                restartOutputPath=os.path.join(self.cc3dSimOutputDir,'restart_'+string.zfill(str(_step),self.__stepNumberOfDigits))# fills string with 0's up to self.__stepNumberOfDigits
                simFilesOutputPath=restartOutputPath
                # one more level of nesting
                restartOutputPath=os.path.join(restartOutputPath,'restart')# fills string with 0's up to self.__stepNumberOfDigits
                
                try:
                    os.makedirs(restartOutputPath)
                except IOError,e:
                    restartOutputPath=''
                    
        else:
            self.cc3dSimOutputDir=self.cc3dSimOutputDir
            restartOutputPath=os.path.join(self.cc3dSimOutputDir,'restart_'+string.zfill(str(_step),self.__stepNumberOfDigits))
            simFilesOutputPath=restartOutputPath
            # one more level of nesting
            restartOutputPath=os.path.join(restartOutputPath,'restart')# fills string with 0's up to self.__stepNumberOfDigits
            
            try:
                os.makedirs(restartOutputPath)
            except IOError,e:
                restartOutputPath=''
                
        # we only copy simulation files if simulation run in in the .cc3d format                
        import re
        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            import CC3DSimulationDataHandler
        
            cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
            #copying  verbatim simulation files
            if not self.__baseSimulationFilesCopied:
                cc3dSimulationDataHandler.copySimulationDataFiles(self.cc3dSimOutputDir)                
                self.__baseSimulationFilesCopied=True
                
            #copying modified simulation files - with restart modification
            if simFilesOutputPath!='':
                cc3dSimulationDataHandler.copySimulationDataFiles(simFilesOutputPath)
                cc3dSimulationDataHandlerLocal=CC3DSimulationDataHandler.CC3DSimulationDataHandler()
                
                simBaseName = os.path.basename(str(CompuCellSetup.simulationFileName))
                #path to newly copied simulation file
                simFullName=os.path.join(simFilesOutputPath,simBaseName)
                #read newly copied simulation file - we will add restart tags to it
                cc3dSimulationDataHandlerLocal.readCC3DFileFormat(simFullName)
                
                print '\n\n\n\n cc3dSimulationDataHandlerLocal.cc3dSimulationData=',cc3dSimulationDataHandlerLocal.cc3dSimulationData
                
                # update simulation size in the XML  in case it has changed during the simulation 
                if cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript!='':
                    print 'cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript=',cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript
                    self.updateXMLScript(cc3dSimulationDataHandlerLocal.cc3dSimulationData.xmlScript)
                elif cc3dSimulationDataHandlerLocal.cc3dSimulationData.pythonScript!='':
                    self.updatePythonScript(cc3dSimulationDataHandlerLocal.cc3dSimulationData.pythonScript)
                
                # if serialize resource exists we only modify it by adding restart simulation element
                if cc3dSimulationDataHandlerLocal.cc3dSimulationData.serializerResource:
                    cc3dSimulationDataHandlerLocal.cc3dSimulationData.serializerResource.restartDirectory='restart'                    
                    cc3dSimulationDataHandlerLocal.writeCC3DFileFormat(simFullName)
                else: # otherwise we create new simulation resource and add restart siukation element
                    cc3dSimulationDataHandlerLocal.cc3dSimulationData.addNewSerializerResource(_restartDir='restart')
                    cc3dSimulationDataHandlerLocal.writeCC3DFileFormat(simFullName)
                
            
            # if self.cc3dSimOutputDir!='':
                # cc3dSimulationDataHandler.copySimulationDataFiles(self.cc3dSimOutputDir)
            
       
        return restartOutputPath
        
    def updatePythonScript(self,_fileName):    
        if _fileName=='':
            return
        
        import re
        dimRegex=re.compile('([\s\S]*.ElementCC3D\([\s]*"Dimensions")([\S\s]*)(\)[\s\S]*)')
        commentRegex=re.compile('^([\s]*#)')
        
        try:
            fXMLNew=open(_fileName+'.new','w')
        except IOerror,e:
            print __file__+' updatePythonScript: could not open ',_fileName,' for writing'
            
        fieldDim=self.sim.getPotts().getCellFieldG().getDim()            
        
        for line in open(_fileName):
            lineTmp= line.rstrip()
            groups=dimRegex.search(lineTmp)            
            
            commentGroups=commentRegex.search(lineTmp)
            if commentGroups:
               
                
                print >>fXMLNew,line.rstrip()
                continue
                
            if groups and groups.lastindex==3:                               
                dimString=',{"x":'+str(fieldDim.x)+',' +'"y":'+str(fieldDim.y)+','+'"z":'+str(fieldDim.z)+'}'                
                newLine=dimRegex.sub(r'\1'+ dimString+r'\3',lineTmp)
                print >>fXMLNew,newLine
            else:            
                print >>fXMLNew,line.rstrip()
            
        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName+'.new',_fileName)
            
        
    def  updateXMLScript(self,_fileName=''):
        if _fileName=='':
            return
        
        import re
        dimRegex=re.compile('([\s]*<Dimensions)([\S\s]*)(/>[\s]*)')
        
        try:
            fXMLNew=open(_fileName+'.new','w')
        except IOerror,e:
            print __file__+' updateXMLScript: could not open ',_fileName,' for writing'
        
        fieldDim=self.sim.getPotts().getCellFieldG().getDim()        
        for line in open(_fileName):
            lineTmp= line.rstrip()
            groups=dimRegex.search(lineTmp)            
            
            if groups and groups.lastindex==3:                
                dimString=' x="'+str(fieldDim.x)+'" '+'y="'+str(fieldDim.y)+'" '+'z="'+str(fieldDim.z)+'" '
                newLine=dimRegex.sub(r'\1'+ dimString+r'\3',lineTmp)
                print >>fXMLNew,newLine
            else:
            
                print >>fXMLNew,line.rstrip()
            
        fXMLNew.close()
        # ged rid of temporary file
        os.remove(_fileName)
        os.rename(_fileName+'.new',_fileName)
        
    def readRestartFile(self,_fileName):
        import XMLUtils
        xml2ObjConverter = XMLUtils.Xml2Obj()

        fileFullPath=os.path.abspath(_fileName)
        
        root_element=xml2ObjConverter.Parse(fileFullPath) # this is RestartFiles element
        if root_element.findAttribute('Version'):
            self.__restartVersion=root_element.getAttribute('Version')
        if root_element.findAttribute('Build'):
            self.__restartVersion=root_element.getAttributeAsInt('Build')
            
        stepElem=root_element.getFirstElement('Step')
        
        if stepElem:
            self.__restartStep=stepElem.getInt()
            
        restartObjectElements=XMLUtils.CC3DXMLListPy(root_element.getElements('ObjectData'))
        
        import SerializerDEPy
        if restartObjectElements:
            for elem in restartObjectElements:
                sd=SerializerDEPy.SerializeData()
                if elem.findAttribute('ObjectName'):
                    sd.objectName=elem.getAttribute('ObjectName')  
                if elem.findAttribute('ObjectType'):
                    sd.objectType=elem.getAttribute('ObjectType')  
                if elem.findAttribute('ModuleName'):
                    sd.moduleName=elem.getAttribute('ModuleName')  
                if elem.findAttribute('ModuleType'):
                    sd.moduleType=elem.getAttribute('ModuleType')  
                if elem.findAttribute('FileName'):
                    sd.fileName=elem.getAttribute('FileName')  
                if elem.findAttribute('FileFormat'):
                    sd.fileFormat=elem.getAttribute('FileFormat')                      
                if sd.objectName!='':    
                    self.__restartResourceDict[sd.objectName]=sd
        print 'self.__restartResourceDict=',self.__restartResourceDict
   
        
    def loadRestartFiles(self):        
        import CompuCellSetup
        import re
        
        print  "\n\n\n\n REASTART MANAGER CompuCellSetup.simulationFileName=",CompuCellSetup.simulationFileName
        
        if re.match(".*\.cc3d$", str(CompuCellSetup.simulationFileName)):
            print "EXTRACTING restartEnabled"
            import CC3DSimulationDataHandler        
            cc3dSimulationDataHandler=CC3DSimulationDataHandler.CC3DSimulationDataHandler()
            cc3dSimulationDataHandler.readCC3DFileFormat(str(CompuCellSetup.simulationFileName))
            print "cc3dSimulationDataHandler.cc3dSimulationData.serializerResource=",cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory
            if cc3dSimulationDataHandler.cc3dSimulationData.serializerResource.restartDirectory!='':
                restartFileLocation=os.path.dirname(str(CompuCellSetup.simulationFileName))
                self.__restartDirectory=os.path.join(restartFileLocation,'restart')
                self.__restartDirectory=os.path.abspath(self.__restartDirectory) # normalizing path format
                
                self.__restartFile=os.path.join(self.__restartDirectory,'restart.xml')
                print 'self.__restartDirectory=',self.__restartDirectory
                print 'self.__restartFile=',self.__restartFile
                self.readRestartFile(self.__restartFile)
                
        #---------------------- LOADING RESTART FILES    --------------------
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
        #---------------------- END OF LOADING RESTART FILES    --------------------        
            
#        
                
        
    def loadCellField(self,):
        import SerializerDEPy
        if 'CellField' in self.__restartResourceDict.keys():
            sd=self.__restartResourceDict['CellField']
            # full path to cell field serialized recource
            fullPath=os.path.join(self.__restartDirectory,sd.fileName)
            fullPath=os.path.abspath(fullPath) # normalizing path format
            tmpFileName=sd.fileName
            sd.fileName=fullPath
            self.serializer.loadCellField(sd)
            sd.fileName=tmpFileName
        
        
    def loadConcentrationFields(self):        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectType=='ConcentrationField':
            
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                tmpFileName=sd.fileName
                sd.fileName=fullPath
                self.serializer.loadConcentrationField(sd)
                sd.fileName=tmpFileName


    def loadScalarFields(self):        
        import CompuCellSetup   
        scalarFieldsDict=CompuCellSetup.fieldRegistry.getScalarFields()        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectType=='ScalarField' and sd.moduleType=='Python':
            
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                tmpFileName=sd.fileName
                sd.fileName=fullPath
                
                try:
                    sd.objectPtr=scalarFieldsDict[sd.objectName]
                    
                except LookupError,e:
                    continue
                
                self.serializer.loadScalarField(sd)
                sd.fileName=tmpFileName
                
    def loadScalarFieldsCellLevel(self):
        
    
        import CompuCellSetup    
        scalarFieldsDictCellLevel=CompuCellSetup.fieldRegistry.getScalarFieldsCellLevel()
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectType=='ScalarFieldCellLevel' and sd.moduleType=='Python':
            
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                tmpFileName=sd.fileName
                sd.fileName=fullPath
                
                

                
                try:
                    sd.objectPtr=scalarFieldsDictCellLevel[sd.objectName]
                    
                except LookupError,e:
                    continue
                
                self.serializer.loadScalarFieldCellLevel(sd)
                sd.fileName=tmpFileName
        
    def loadVectorFields(self): 
        import CompuCellSetup   
        vectorFieldsDict=CompuCellSetup.fieldRegistry.getVectorFields()
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectType=='VectorField' and sd.moduleType=='Python':
            
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                tmpFileName=sd.fileName
                sd.fileName=fullPath
                
                try:
                    sd.objectPtr=vectorFieldsDict[sd.objectName]
                    
                except LookupError,e:
                    continue
                
                self.serializer.loadVectorField(sd)
                sd.fileName=tmpFileName

                
    def loadVectorFieldsCellLevel(self):             
        import CompuCellSetup   
        vectorFieldsCellLevelDict=CompuCellSetup.fieldRegistry.getVectorFieldsCellLevel()
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectType=='VectorFieldCellLevel' and sd.moduleType=='Python':
            
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                tmpFileName=sd.fileName
                sd.fileName=fullPath
                
                try:
                    sd.objectPtr=vectorFieldsCellLevelDict[sd.objectName]
                    
                except LookupError,e:
                    continue
                
                self.serializer.loadVectorFieldCellLevel(sd)
                sd.fileName=tmpFileName
                
    def loadCoreCellAttributes(self):
        import cPickle
        from PySteppables import CellList        
        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='CoreCellAttributes' and sd.objectType=='Pickle'        :


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    clusterId=cell.clusterId
                    # print 'cellId=',cellId
                    cellCoreAttributes=cPickle.load(pf)
                    # print 'cellCoreAttributes=',cellCoreAttributes
                    # cellLocal=inventory.getCellByIds(cellId,clusterId)   
                    
                    if cell:    
                        # print 'cell=',cell," cell.id=",cell.id    
                        self.setCellCoreAttributes(cell, cellCoreAttributes)
                    
                pf.close()
                
                
                
    def unpickleDict(self,_fileName,_cellList):
        import CompuCell
        import cPickle
        import copy
        try:
            pf=open(_fileName,'r')
        except IOError,e:
            return
        
        numberOfCells=cPickle.load(pf)
        
        for cell in _cellList:
            
            cellId=cPickle.load(pf)
            
            unpickledAttribDict=cPickle.load(pf)
            
            dictAttrib=CompuCell.getPyAttrib(cell)
            
            # dictAttrib=copy.deepcopy(unpickledAttribDict)
            dictAttrib.update(unpickledAttribDict) # adds all objects from unpickledAttribDict to dictAttrib -  note: deep copy will not work here
            
        pf.close()
        
    def unpickleList(self,_fileName,_cellList):
        import CompuCell
        import cPickle
        
        try:
            pf=open(_fileName,'r')
        except IOError,e:
            return
        
        numberOfCells=cPickle.load(pf)
        
        for cell in _cellList:
            # print 'cell.id=',cell.id
            cellId=cPickle.load(pf)
            unpickledAttribList=cPickle.load(pf)
            listAttrib=CompuCell.getPyAttrib(cell)            
            listAttrib.extend(unpickledAttribList) # appends all elements of unpickledAttribList to the end of listAttrib -  note: deep copy will not work here
            # print 'listAttrib=',listAttrib
                        
        pf.close()
        
        
    def loadPythonAttributes(self):
        
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='PythonAttributes' and sd.objectType=='Pickle':
    
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                
                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        
                
                # checking if cells have extra attribute
                import CompuCell
                for cell in cellList:             
                    if not CompuCell.isPyAttribValid(cell):
                        return

                listFlag=True
                for cell in cellList:             
                    attrib=CompuCell.getPyAttrib(cell)
                    if isinstance(attrib,list):
                        listFlag=True
                    else:
                        listFlag=False    
                    break
                
                if listFlag:
                    self.unpickleList(fullPath,cellList)
                else:
                    self.unpickleDict(fullPath,cellList)
                    
    def loadSBMLSolvers(self):
        
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        
        # loading and initializing freeFloating SBML Simulators  - SBML solvers associated with cells are loaded (but not fully initialized) in the loadPythonAttributes
        for resourceName, sd in self.__restartResourceDict.iteritems():
            print 'resourceName=',resourceName
            print 'sd=',sd
        
            if sd.objectName=='FreeFloatingSBMLSolvers' and sd.objectType=='Pickle':
                print 'RESTORING FreeFloatingSBMLSolvers '
                
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                with open(fullPath,'r') as pf:                
                    CompuCellSetup.freeFloatingSBMLSimulator=cPickle.load(pf)
                    
                # initializing  freeFloating SBML Simulators       
                for modelName, sbmlSolver in CompuCellSetup.freeFloatingSBMLSimulator.iteritems():
                    sbmlSolver.loadSBML(_externalPath=self.sim.getBasePath())
            
        # full initializing SBML solvers associated with cell  - we do that regardless whether we have freeFloatingSBMLSolver pickled file or not
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)                        

        # checking if cells have extra attribute
        import CompuCell
        for cell in cellList:             
            if not CompuCell.isPyAttribValid(cell):
                return
            else:    
                attrib=CompuCell.getPyAttrib(cell)
                if isinstance(attrib,list):
                    return
                else:
                    break
        
        for cell in cellList:             
        
            cellDict=CompuCell.getPyAttrib(cell)
            try:
                sbmlDict=cellDict['SBMLSolver']
                print 'sbmlDict=',sbmlDict
            except LookupError,e:
                continue
                
            for modelName,sbmlSolver in  sbmlDict.iteritems():   
                sbmlSolver.loadSBML(_externalPath=self.sim.getBasePath()) # this call fully initializes SBML Solver by loading sbmlMode ( relative path stored in sbmlSolver.path and root dir is passed using self.sim.getBasePath())
                        


    def loadBionetSolver(self):
        
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        
        
        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='BionetSolver' and sd.objectType=='Pickle':
    
                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                
                try:
                    pf=open(fullPath,'r')
                except IOError,e:
                    return        
                
                # first will load sbml files and 
                import bionetAPI
                sbmlModelDict=cPickle.load(pf)

                for modelName, modelDict in sbmlModelDict.iteritems():
                    bionetAPI.loadSBMLModel(modelName, modelDict["ModelPath"], modelDict["ModelKey"], modelDict["ModelTimeStep"])


                #loading library names (model names)  associated with cell types
                cellTemplateLibraryDict=cPickle.load(pf)                
                for templateLibraryName, modelNames in  cellTemplateLibraryDict.iteritems(): # templateLibraryName in this case is the sdame as cell type name (except medium)
                    for modelName in modelNames:
                        bionetAPI.addSBMLModelToTemplateLibrary(modelName,templateLibraryName)

                    
                
                nonCellTemplateLibraryDict=cPickle.load(pf)
                # print "nonCellTemplateLibraryDict=",nonCellTemplateLibraryDict
                for nonCellLibName, modelInstanceDict in nonCellTemplateLibraryDict.iteritems():
                    for modelName, varDict in modelInstanceDict.iteritems():
                        bionetAPI.addSBMLModelToTemplateLibrary(modelName,nonCellLibName)
                
                bionetAPI.initializeBionetworks()
                
                # after bionetworks are initialized inthe bionetAPI we can assign variables to non cell models
                for nonCellLibName, modelInstanceDict in nonCellTemplateLibraryDict.iteritems():
                    for modelName, varDict in modelInstanceDict.iteritems():                        
                        for varName, varValue in varDict.iteritems():
                            bionetAPI.setBionetworkValue(varName, varValue,nonCellLibName)
                            
                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        
                
                # checking if cells have extra attribute
                import CompuCell
                for cell in cellList:

                    dictAttrib=CompuCell.getPyAttrib(cell)
                    dictToPickle={} 
                    
                    id=cPickle.load(pf)       # cell id
                    cellSBMLModelData=cPickle.load(pf) # cell's sbml models   
                    
                    for modelName,modeVarDict in cellSBMLModelData.iteritems():
                        for varName,varValue in modeVarDict.iteritems():
                            bionetAPI.setBionetworkValue(varName, varValue,cell.id)
                    
                pf.close()

    def loadAdhesionFlex(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #AdhesionFlexPlugin 
        adhesionFlexPlugin=None
        if self.sim.pluginManager.isLoaded("AdhesionFlex"):
            import CompuCell            
            adhesionFlexPlugin=CompuCell.getAdhesionFlexPlugin()
        else:
            return

        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='AdhesionFlex' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                # read medium adhesion molecule vector
                mediumAdhesionVector=cPickle.load(pf)
                # print 'mediumAdhesionVector=',mediumAdhesionVector
                
                adhesionFlexPlugin.assignNewMediumAdhesionMoleculeDensityVector(mediumAdhesionVector)
                # print "from plugin = ",adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    cellAdhesionVector=cPickle.load(pf)
                    adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(cell,cellAdhesionVector)

                pf.close()
            adhesionFlexPlugin.overrideInitialization()  
        # sys.exit()

    def loadChemotaxis(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #chemotaxisPlugin 
        chemotaxisPlugin=None
        if self.sim.pluginManager.isLoaded("Chemotaxis"):
            import CompuCell            
            chemotaxisPlugin=CompuCell.getChemotaxisPlugin()
        else:
            return
        
        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='Chemotaxis' and sd.objectType=='Pickle'        :

                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    #loading number of chemotaxis data that cell has
                    chdNumber=cPickle.load(pf)      
                    for i in range(chdNumber):
                        # reading chemotaxis data 
                        chdDict=cPickle.load(pf)      
                        #creating chemotaxis data for cell 
                        chd=chemotaxisPlugin.addChemotaxisData(cell,chdDict['fieldName'])
                        chd.setLambda(chdDict['lambda'])
                        chd.saturationCoef=chdDict['saturationCoef']
                        chd.setChemotaxisFormulaByName(chdDict['formulaName'])
                        chd.assignChemotactTowardsVectorTypes(chdDict['chemotactTowardsTypesVec'])

                pf.close()
                
                
    def loadLengthConstraint(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #LengthConstraintPlugin 
        lengthConstraintPlugin=None        
        if self.sim.pluginManager.isLoaded("LengthConstraint"):
            import CompuCell            
            lengthConstraintPlugin=CompuCell.getLengthConstraintPlugin()            
        else:
            return

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='LengthConstraint' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    lengthConstraintVec=cPickle.load(pf)
                    # ([lcp.getLambdaLength(cell),lcp.getTargetLength(cell),lcp.getMinorTargetLength(cell)],pf)
                    lengthConstraintPlugin.setLengthConstraintData(cell,lengthConstraintVec[0],lengthConstraintVec[1],lengthConstraintVec[2])        

                pf.close()
            
    def loadConnectivityGlobal(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #ConnectivityLocalFlexPlugin    
        connectivityGlobalPlugin=None        
        if self.sim.pluginManager.isLoaded("ConnectivityGlobal"):
            import CompuCell            
            connectivityGlobalPlugin=CompuCell.getConnectivityGlobalPlugin()            
        else:
            return

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='ConnectivityGlobal' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    connectivityStrength=cPickle.load(pf)                    
                    connectivityGlobalPlugin.setConnectivityStrength(cell,connectivityStrength)

                pf.close()


    def loadConnectivityLocalFlex(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #ConnectivityLocalFlexPlugin    
        connectivityLocalFlexPlugin=None        
        if self.sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            import CompuCell            
            connectivityLocalFlexPlugin=CompuCell.getConnectivityLocalFlexPlugin()            
        else:
            return

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='ConnectivityLocalFlex' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    connectivityStrength=cPickle.load(pf)                    
                    connectivityLocalFlexPlugin.setConnectivityStrength(cell,connectivityStrength)

                pf.close()


                
                
    def loadFocalPointPlasticity(self):
    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
                
        #FocalPointPlasticity
        focalPointPlasticityPlugin=None
        if self.sim.pluginManager.isLoaded("FocalPointPlasticity"):
            import CompuCell
            focalPointPlasticityPlugin=CompuCell.getFocalPointPlasticityPlugin()         
        else:
            return

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='FocalPointPlasticity' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)

                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    cellId=cell.id
                    clusterId=cell.clusterId
                    
                    # read number of fpp links in the cell (external)
                    linksNumber=cPickle.load(pf)
                    for i in range(linksNumber):
                        fppDict=cPickle.load(pf) # loading external links
                        fpptd=CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        neighborIds=fppDict['neighborIds'] # cellId, cluster id
                        
                        neighborCell=inventory.getCellByIds(neighborIds[0],neighborIds[1])
                        fpptd.neighborAddress=neighborCell
                        fpptd.lambdaDistance=fppDict['lambdaDistance']
                        fpptd.targetDistance=fppDict['targetDistance']
                        fpptd.maxDistance=fppDict['maxDistance']
                        fpptd.activationEnergy=fppDict['activationEnergy']
                        fpptd.maxNumberOfJunctions=fppDict['maxNumberOfJunctions']
                        fpptd.neighborOrder=fppDict['neighborOrder']
                        
                        focalPointPlasticityPlugin.insertFPPData(cell,fpptd)
                        
                    # read number of fpp links in the cell (internal)
                    internalLinksNumber=cPickle.load(pf)
                    for i in range(internalLinksNumber):
                        fppDict=cPickle.load(pf) # loading external links
                        fpptd=CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        neighborIds=fppDict['neighborIds'] # cellId, cluster id
                        neighborCell=inventory.getCellByIds(neighborIds[0],neighborIds[1])
                        fpptd.neighborAddfess=neighborCell
                        fpptd.lambdaDistance=fppDict['lambdaDistance']
                        fpptd.targetDistance=fppDict['targetDistance']
                        fpptd.maxDistance=fppDict['maxDistance']
                        fpp.activationEnergy=fppDict['activationEnergy']
                        fpptd.maxNumberOfJunctions=fppDict['maxNumberOfJunctions']
                        fpptd.neighborOrder=fppDict['neighborOrder']
                        focalPointPlasticityPlugin.insertInternalFPPData(cell,fpptd)
                        
                    # read number of fpp links in the cell (anchors)
                    anchorLinksNumber=cPickle.load(pf)
                    for i in range(anchorLinksNumber):
                        fppDict=cPickle.load(pf) # loading external links
                        fpptd=CompuCell.FocalPointPlasticityTrackerData()
                        # get neighbor data
                        # neighborIds=fppDict['neighborIds'] # cellId, cluster id
                        # neighborCell=inventory.getCellByIds(neighborIds[0],neighborIds[1])
                        fpptd.neighborAddfess=0
                        fpptd.lambdaDistance=fppDict['lambdaDistance']
                        fpptd.targetDistance=fppDict['targetDistance']
                        fpptd.maxDistance=fppDict['maxDistance']                        
                        fpptd.anchorId=fppDict['anchorId']
                        fpptd.anchorPoint[0]=fppDict['anchorPoint'][0]
                        fpptd.anchorPoint[1]=fppDict['anchorPoint'][1]
                        fpptd.anchorPoint[2]=fppDict['anchorPoint'][2]
                        
                        
                        focalPointPlasticityPlugin.insertAnchorFPPData(cell,fpptd)

                        

                pf.close()
 


    def loadContactLocalProduct(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #ContactLocalProductPlugin 
        contactLocalProductPlugin=None
        if self.sim.pluginManager.isLoaded("ContactLocalProduct"):
            import CompuCell            
            contactLocalProductPlugin=CompuCell.getContactLocalProductPlugin()         
        else:
            return

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='ContactLocalProduct' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)
                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)                    
                    cadherinVector=cPickle.load(pf)      
                    contactLocalProductPlugin.setCadherinConcentrationVec(cell,CompuCell.contactproductdatacontainertype(cadherinVector))                    

                pf.close()

                
    def loadCellOrientation(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #CellOrientationPlugin 
        cellOrientationPlugin=None
        if self.sim.pluginManager.isLoaded("CellOrientation"):
            import CompuCell            
            cellOrientationPlugin=CompuCell.getCellOrientationPlugin()    
        else:
            return
            

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='CellOrientation' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    lambdaCellOrientation=cPickle.load(pf)
                    cellOrientationPlugin.setLambdaCellOrientation(cell,lambdaCellOrientation)

                pf.close()
                

    def loadPolarizationVector(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #PolarizationVectorPlugin 
        polarizationVectorPlugin=None
        if self.sim.pluginManager.isLoaded("PolarizationVector"):
            import CompuCell            
            polarizationVectorPlugin=CompuCell.getPolarizationVectorPlugin()
        else:
            return
            

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='PolarizationVector' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    polarizationVec=cPickle.load(pf)
                    polarizationVectorPlugin.setPolarizationVector(cell,polarizationVec[0],polarizationVec[1],polarizationVec[2])
                    

                pf.close()         


    def outputPolarization23Plugin(self,_restartOutputPath,_rstXMLElem):
    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #polarization23Plugin 
        polarization23Plugin=None
        if self.sim.pluginManager.isLoaded("Polarization23"):
            import CompuCell            
            polarization23Plugin=CompuCell.getPolarization23Plugin()
        else:    
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Polarization23'
        sd.moduleType='Plugin'
        sd.objectName='Polarization23'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'Polarization23'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            polVec=polarization23Plugin.getPolarizationVector(cell)            
            cPickle.dump([polVec.fX,polVec.fY,polVec.fZ],pf)
            cPickle.dump(polarization23Plugin.getPolarizationMarkers(),pf)
            cPickle.dump(polarization23Plugin.getLambdaPolarization(),pf)
                    
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)
    
                
    def loadPolarization23(self):

        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        
        #polarization23Plugin 
        polarization23Plugin=None
        if self.sim.pluginManager.isLoaded("Polarization23"):
            import CompuCell            
            polarization23Plugin=CompuCell.getPolarization23Plugin()
        else:    
            return
            

        for resourceName, sd in self.__restartResourceDict.iteritems():
            if sd.objectName=='Polarization23' and sd.objectType=='Pickle':


                inventory = self.sim.getPotts().getCellInventory()                
                cellList = CellList(inventory)        

                fullPath=os.path.join(self.__restartDirectory,sd.fileName)
                fullPath=os.path.abspath(fullPath) # normalizing path format
                try:
                    pf=open(fullPath,'r')                
                except IOError,e:
                    return
                
                numberOfCells=cPickle.load(pf)                
                
                for cell in cellList:            
                    cellId=cPickle.load(pf)
                    
                    polVec=cPickle.load(pf) #[fX,fY,fZ]
                    polMarkers=cPickle.load(pf)
                    lambdaPol=cPickle.load(pf)
                    
                    
                    polarization23Plugin.setPolarizationVector(cell,CompuCell.Vector3(polVec[0],polVec[2],polVec[2]))
                    polarization23Plugin.setPolarizationMarkers(cell,polMarkers[0],polMarkers[1])
                    polarization23Plugin.setLambdaPolarization(cell,lambdaPol)

                pf.close()         

                
                
    def outputRestartFiles(self,_step=0,_onDemand=False):
        
        if not _onDemand and self.__outputFrequency<=0:
            return
        
        if not _onDemand and _step==0:
            return
        
        if not _onDemand and _step%self.__outputFrequency:            
            return 
        
        self.serializer.init(self.sim) # have to initialize serialized each time in case lattice gets resized in which case cellField Ptr has to be updated and lattice dimension is usually different
        
        from XMLUtils import ElementCC3D
        import Version
        rstXMLElem=ElementCC3D("RestartFiles",{"Version":Version.getVersionAsString(),'Build':Version.getSVNRevisionAsString()})
        rstXMLElem.ElementCC3D("Step",{},_step)
        print 'outputRestartFiles'
        import CompuCellSetup
        cc3dSimOutputDir=CompuCellSetup.screenshotDirectoryName
        print "cc3dSimOutputDir=",cc3dSimOutputDir
        
        print "CompuCellSetup.simulationPaths.simulationXMLFileName=",CompuCellSetup.simulationPaths.simulationXMLFileName
        print 'CompuCellSetup.simulationFileName=',CompuCellSetup.simulationFileName
        restartOutputPath=self.setupRestartOutputDirectory(_step)
        
        
        
        
        if restartOutputPath=='':
            return # no output if restartOutputPath is not specified
            
        #---------------------- OUTPUTTING RESTART FILES    --------------------
        # outputting cell field    
        self.outputCellField(restartOutputPath,rstXMLElem)        
        # outputting concentration fields (scalar fields) from PDE solvers    
        self.outputConcentrationFields(restartOutputPath,rstXMLElem)
        # outputting extra scalar fields   - used in Python only
        self.outputScalarFields(restartOutputPath,rstXMLElem)
        # outputting extra scalar fields cell level  - used in Python only
        self.outputScalarFieldsCellLevel(restartOutputPath,rstXMLElem)
        # outputting extra vector fields  - used in Python only
        self.outputVectorFields(restartOutputPath,rstXMLElem)        
        # outputting extra vector fields cell level  - used in Python only
        self.outputVectorFieldsCellLevel(restartOutputPath,rstXMLElem)        
        # outputting core cell  attributes
        self.outputCoreCellAttributes(restartOutputPath,rstXMLElem)       
        # outputting cell Python attributes
        self.outputPythonAttributes(restartOutputPath,rstXMLElem)                     
        # outputting bionetSolver
        self.outputBionetSolver(restartOutputPath,rstXMLElem)          
        # outputting FreeFloating SBMLSolvers - notice that SBML solvers assoaciated with a cell are pickled in the outputPythonAttributes function
        self.outputFreeFloatingSBMLSolvers(restartOutputPath,rstXMLElem)        
        # outputting plugins
        # outputting AdhesionFlexPlugin
        self.outputAdhesionFlexPlugin(restartOutputPath,rstXMLElem)    
        # outputting ChemotaxisPlugin
        self.outputChemotaxisPlugin(restartOutputPath,rstXMLElem)    
        # outputting LengthConstraintPlugin
        self.outputLengthConstraintPlugin(restartOutputPath,rstXMLElem)    
        # outputting ConnectivityGlobalPlugin
        self.outputConnectivityGlobalPlugin(restartOutputPath,rstXMLElem)    
        # outputting ConnectivityLocalFlexPlugin
        self.outputConnectivityLocalFlexPlugin(restartOutputPath,rstXMLElem)    
        # outputting FocalPointPlacticityPlugin
        self.outputFocalPointPlacticityPlugin(restartOutputPath,rstXMLElem)
        # outputting ContactLocalProductPlugin
        self.outputContactLocalProductPlugin(restartOutputPath,rstXMLElem)
        # outputting CellOrientationPlugin
        self.outputCellOrientationPlugin(restartOutputPath,rstXMLElem)
        # outputting PolarizationVectorPlugin
        self.outputPolarizationVectorPlugin(restartOutputPath,rstXMLElem)
        # outputting Polarization23Plugin
        self.outputPolarization23Plugin(restartOutputPath,rstXMLElem)

        
        #---------------------- END OF  OUTPUTTING RESTART FILES    --------------------               
        
        #-------------writing xml description of the restart files
        rstXMLElem.CC3DXMLElement.saveXML(os.path.join(restartOutputPath,'restart.xml'))
        
        #--------------- depending on removePreviousFiles we will remove or keep previous restart files
        print '\n\n\n\n self.__allowMultipleRestartDirectories=',self.__allowMultipleRestartDirectories
        if not self.__allowMultipleRestartDirectories:
        
            print '\n\n\n\n self.__completedRestartOutputPath=',self.__completedRestartOutputPath
            
            if self.__completedRestartOutputPath!='':
                import shutil
                try:
                    shutil.rmtree(self.__completedRestartOutputPath)
                except:
                    # will ignore exceptions during directory removal - they might be due e.g. user accessing directory to be removed - in such a case it is best to ignore such requests
                    pass
                    
                
        self.__completedRestartOutputPath=self.getRestartOutputRootPath(restartOutputPath)
        
    def outputConcentrationFields(self,_restartOutputPath,_rstXMLElem):
        concFieldNameVec=self.sim.getConcentrationFieldNameVector()
        import SerializerDEPy
        for fieldName in concFieldNameVec:
            sd=SerializerDEPy.SerializeData()
            sd.moduleName='PDESolver'
            sd.moduleType='Steppable'
            
            sd.objectName=fieldName
            sd.objectType='ConcentrationField'
            sd.fileName=os.path.join(_restartOutputPath,fieldName+'.dat')
            print 'sd.fileName=',sd.fileName
            sd.fileFormat='text'
            self.serializeDataList.append(sd)
            self.serializer.serializeConcentrationField(sd)
            self.appendXMLStub(_rstXMLElem,sd)
            print "Got concentration field: ",fieldName
            
    def outputCellField(self,_restartOutputPath,_rstXMLElem):
        concFieldNameVec=self.sim.getConcentrationFieldNameVector()
        import SerializerDEPy
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Potts3D'
        sd.moduleType='Core'
        
        sd.objectName='CellField'
        sd.objectType='CellField'
        sd.fileName=os.path.join(_restartOutputPath,sd.objectName+'.dat')        
        sd.fileFormat='text'
        self.serializeDataList.append(sd)
        self.serializer.serializeCellField(sd)
        self.appendXMLStub(_rstXMLElem,sd)
        
    def outputScalarFields(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup
        scalarFieldsDict=CompuCellSetup.fieldRegistry.getScalarFields()
        for fieldName in scalarFieldsDict:
            sd=SerializerDEPy.SerializeData()
            sd.moduleName='Python'
            sd.moduleType='Python'
            sd.objectName=fieldName
            sd.objectType='ScalarField'
            sd.objectPtr=scalarFieldsDict[fieldName]
            sd.fileName=os.path.join(_restartOutputPath,fieldName+'.dat')
            self.serializer.serializeScalarField(sd)
            self.appendXMLStub(_rstXMLElem,sd)
            
    def outputScalarFieldsCellLevel(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup
        scalarFieldsDictCellLevel=CompuCellSetup.fieldRegistry.getScalarFieldsCellLevel()
        for fieldName in scalarFieldsDictCellLevel:
            sd=SerializerDEPy.SerializeData()
            sd.moduleName='Python'
            sd.moduleType='Python'
            sd.objectName=fieldName
            sd.objectType='ScalarFieldCellLevel'
            sd.objectPtr=scalarFieldsDictCellLevel[fieldName]
            sd.fileName=os.path.join(_restartOutputPath,fieldName+'.dat')
            self.serializer.serializeScalarFieldCellLevel(sd)
            self.appendXMLStub(_rstXMLElem,sd)
        
            
    def outputVectorFields(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup
        vectorFieldsDict=CompuCellSetup.fieldRegistry.getVectorFields()
        for fieldName in vectorFieldsDict:
            sd=SerializerDEPy.SerializeData()
            sd.moduleName='Python'
            sd.moduleType='Python'
            sd.objectName=fieldName
            sd.objectType='VectorField'
            sd.objectPtr=vectorFieldsDict[fieldName]
            sd.fileName=os.path.join(_restartOutputPath,fieldName+'.dat')
            self.serializer.serializeVectorField(sd)
            self.appendXMLStub(_rstXMLElem,sd)
            
    def outputVectorFieldsCellLevel(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup
        vectorFieldsCellLevelDict=CompuCellSetup.fieldRegistry.getVectorFieldsCellLevel()
        for fieldName in vectorFieldsCellLevelDict:
            sd=SerializerDEPy.SerializeData()
            sd.moduleName='Python'
            sd.moduleType='Python'
            sd.objectName=fieldName
            sd.objectType='VectorFieldCellLevel'
            sd.objectPtr=vectorFieldsCellLevelDict[fieldName]
            sd.fileName=os.path.join(_restartOutputPath,fieldName+'.dat')
            self.serializer.serializeVectorFieldCellLevel(sd)
            self.appendXMLStub(_rstXMLElem,sd)
            
    def cellCoreAttributes(self,_cell):
        coreAttribDict={}
        coreAttribDict['targetVolume']=_cell.targetVolume
        coreAttribDict['lambdaVolume']=_cell.lambdaVolume
        coreAttribDict['targetSurface']=_cell.targetSurface
        coreAttribDict['lambdaSurface']=_cell.lambdaSurface
        coreAttribDict['targetClusterSurface']=_cell.targetClusterSurface
        coreAttribDict['lambdaClusterSurface']=_cell.lambdaClusterSurface
        coreAttribDict['type']=_cell.type
        coreAttribDict['xCOMPrev']=_cell.xCOMPrev
        coreAttribDict['yCOMPrev']=_cell.yCOMPrev
        coreAttribDict['zCOMPrev']=_cell.zCOMPrev
        coreAttribDict['lambdaVecX']=_cell.lambdaVecX
        coreAttribDict['lambdaVecY']=_cell.lambdaVecY
        coreAttribDict['lambdaVecZ']=_cell.lambdaVecZ
        coreAttribDict['flag']=_cell.flag
        coreAttribDict['fluctAmpl']=_cell.fluctAmpl
        
        return coreAttribDict

    def setCellCoreAttributes(self,_cell, _coreAttribDict):
    
        for attribName,attribValue in _coreAttribDict.iteritems():
            
            try:    
                setattr(_cell,attribName,attribValue)
                
            except LookupError,e:            
                continue
            except AttributeError,ea:                            
                continue
                
        
        
    def outputCoreCellAttributes(self,_restartOutputPath,_rstXMLElem):    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Potts3D'
        sd.moduleType='Core'
        sd.objectName='CoreCellAttributes'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'CoreCellAttributes'+'.dat')
        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        for cell in cellList:            
            cPickle.dump(cell.id,pf)
            cPickle.dump(self.cellCoreAttributes(cell),pf)        
            
            
        pf.close()
        self.appendXMLStub(_rstXMLElem,sd)
    
    def pickleList(self,_fileName,_cellList):
        import CompuCell
        import cPickle
        
        numberOfCells = len(_cellList)
        
        
        nullFile=open(os.devnull,'w')
        try:
            pf=open(_fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in _cellList:
            # print 'cell.id=',cell.id
            listAttrib=CompuCell.getPyAttrib(cell)
            listToPickle=[]
            # checking which list items are picklable
            for item in listAttrib:
                try:
                    cPickle.dump(item,nullFile)
                    listToPickle.append(item)
                except TypeError,e:
                    print "PICKLNG LIST"
                    print e
                    pass
                    
            cPickle.dump(cell.id,pf)
            cPickle.dump(listToPickle,pf)        
            
        nullFile.close()
        pf.close()        

    def pickleDictionary(self,_fileName,_cellList):
        import CompuCell
        import cPickle
        
        numberOfCells = len(_cellList)
        
        
        nullFile=open(os.devnull,'w')
        try:
            pf=open(_fileName,'w')
        except IOError,e:
            return
            
        
        #--------------------------
        # pt=CompuCell.Vector3(10,11,12)
        
        # pf1=open('PickleCC3D.dat','w')
        # cPickle.dump(pt,pf1)
                
        # pf1.close()
        
        # pf1=open('PickleCC3D.dat','r')
        
        
        # content=cPickle.load(pf1)
        # print 'content=',content
        # print 'type(content)=',type(content)
        # pf1.close()
        #--------------------------
        
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in _cellList:
            # print 'cell.id=',cell.id
            dictAttrib=CompuCell.getPyAttrib(cell)
            dictToPickle={} 
            # checking which list items are picklable
            for key in dictAttrib:
                try:
                    cPickle.dump(dictAttrib[key],nullFile)
                    dictToPickle[key]=dictAttrib[key]
                    
                except TypeError,e:
                    print "key=",key," cannot be pickled"
                    print e
                    pass
                    
            cPickle.dump(cell.id,pf)
            cPickle.dump(dictToPickle,pf)        
            
        nullFile.close()
        pf.close()        

    def outputFreeFloatingSBMLSolvers(self,_restartOutputPath,_rstXMLElem): 
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Python'
        sd.moduleType='Python'
        sd.objectName='FreeFloatingSBMLSolvers'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'FreeFloatingSBMLSolvers'+'.dat')
        if CompuCellSetup.freeFloatingSBMLSimulator: # checking if freeFloatingSBMLSimulator is non-empty        
            with open(sd.fileName,'w') as pf:
                cPickle.dump(CompuCellSetup.freeFloatingSBMLSimulator,pf) 
                self.appendXMLStub(_rstXMLElem,sd)
                
    
    def outputPythonAttributes(self,_restartOutputPath,_rstXMLElem):
        # notice that this function also outputs SBMLSolver objects
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        
        # checking if cells have extra attribute
        import CompuCell
        for cell in cellList:             
            if not CompuCell.isPyAttribValid(cell):
                return

        listFlag=True
        for cell in cellList:             
            attrib=CompuCell.getPyAttrib(cell)
            if isinstance(attrib,list):
                listFlag=True
            else:
                listFlag=False    
            break    
        
        print 'listFlag=',listFlag
        
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Python'
        sd.moduleType='Python'
        sd.objectName='PythonAttributes'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'PythonAttributes'+'.dat')
        # cPickle.dump(numberOfCells,pf)
        
        if listFlag:
            self.pickleList(sd.fileName,cellList)
        else:
            self.pickleDictionary(sd.fileName,cellList)
        
        self.appendXMLStub(_rstXMLElem,sd)
        
    def outputBionetSolver(self,_restartOutputPath,_rstXMLElem):
        
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        try: # some Cc3D distros might be without bionetSolver
            import bionetAPI
            
            
            if bionetAPI.bionetworkManager is None:
                print "\t\t\t bionetAPI.bionetworkManager is not initialized"
                return
            else:
                bbm=bionetAPI.bionetworkManager
            
            
        except ImportError,e:
            return
            
        # checking if cells have extra attribute
        import CompuCell
        for cell in cellList:             
            if not CompuCell.isPyAttribValid(cell):
                return
                
        listFlag=True
        for cell in cellList:             
            attrib=CompuCell.getPyAttrib(cell)
            if isinstance(attrib,list):
                listFlag=True
            else:
                listFlag=False    
            break    

        print "CELLS HAVE ATTRIBUTES"                    
        print 'listFlag=',listFlag

        import shutil        
        sbmlModelDict={}
        for modelName,model in bbm.bionetworkSBMLInventory.iteritems():
            print "modelName=",modelName," integrationStep=",model.getTimeStepSize()," path=",os.path.basename(model.getModelPath())," model Key=",model.getModelKey()
            sbmlModelDict[modelName]={"ModelKey":model.getModelKey(),"ModelTimeStep":model.getTimeStepSize(),"ModelPath":os.path.join("Simulation",os.path.basename(model.getModelPath()))}    
            #just in case - copy all sbml files to Simulation directory of in the output folder

            sbmlModelOutputPath = os.path.join(self.cc3dSimOutputDir,'Simulation')
            
            if not os.path.exists(sbmlModelOutputPath):
                
                os.mkdir(sbmlModelOutputPath)
                
            #copy project file        
            try:            
                shutil.copy(model.getModelPath(),sbmlModelOutputPath) 
            except: # ignore any copy errors
                pass
            
                        
        
        
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='BionetSolver'
        sd.moduleType='Python'
        sd.objectName='BionetSolver'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'BionetSolver'+'.dat')
        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return      
        
        
        cPickle.dump(sbmlModelDict,pf)
        
        
        # cPickle.dump(len(bbm.nonCellBionetworkInventory.keys()),pf) # dumping number of nonCell BionetworkTemplateLibraries
        nonCellTemplateLibraryDict={}
        
        nonCellLibraryNames=bbm.nonCellBionetworkInventory.keys()
        
        for templateLibraryName,bn in bbm.nonCellBionetworkInventory.iteritems():
            dictToPickle={}
            templateLibrary=bn.getTemplateLibraryInstancePtr()
            
            modelNames=templateLibrary.getModelNamesAsString().split()
            
            for name in modelNames:
                stateVarNames=bn.getBionetworkStateVarNamesAsString(name)
                listOfStateVars=stateVarNames.split()
            
                model=templateLibrary.getSBMLModelByName(name)
                
                modelKey=sbmlModelDict[name]["ModelKey"]
                modelStateVarDict={}
                for stateVar in listOfStateVars:
                    varAccessName=modelKey+"_"+stateVar
                    modelStateVarDict[varAccessName]=bionetAPI.getBionetworkValue(varAccessName , templateLibraryName)
                    
                dictToPickle[name]= modelStateVarDict
            
            nonCellTemplateLibraryDict[templateLibraryName]=dictToPickle
            
            
        # we store nonCell Template Library data after  template library data associated with cell types
        
        # templateLibraries associated with cell types
        cellTemplateLibraryDict={}
        
        for templateLibraryName,templateLibrary in bbm.bionetworkTemplateLibraryInventory.iteritems():         
            dictToPickle={}
            if templateLibraryName not in nonCellLibraryNames:
                modelNames=templateLibrary.getModelNamesAsString().split()
                # print "templateLibraryName=",templateLibraryName, "\t\t\t templateLibrary.modelNames=",templateLibrary.getModelNamesAsString().split()
                cellTemplateLibraryDict[templateLibraryName]=templateLibrary.getModelNamesAsString().split()
            
            
        cPickle.dump(cellTemplateLibraryDict,pf) # we first store template library data associated with cell types
        
        cPickle.dump(nonCellTemplateLibraryDict,pf) # later we store nonCell Template Library data

            
        
        for cell in cellList:
            
            dictAttrib=CompuCell.getPyAttrib(cell)
            dictToPickle={} 
            # checking which list items are picklable
            try:
            
                bn=dictAttrib["Bionetwork"]
                
                stateVarNames=bn.getBionetworkStateVarNamesAsString("DeltaNotch")
                listOfStateVars=stateVarNames.split()
                
                templateLibrary=bn.getTemplateLibraryInstancePtr()
                modelNames=templateLibrary.getModelNamesAsString().split()
                
                for name in modelNames:
                    stateVarNames=bn.getBionetworkStateVarNamesAsString(name)
                    listOfStateVars=stateVarNames.split()

                    model=templateLibrary.getSBMLModelByName(name)
                    
                    modelKey=sbmlModelDict[name]["ModelKey"]
                    modelStateVarDict={}
                    for stateVar in listOfStateVars:
                        varAccessName=modelKey+"_"+stateVar
                        modelStateVarDict[varAccessName]=bionetAPI.getBionetworkValue(varAccessName,cell.id)
                        
                    dictToPickle[name]= modelStateVarDict

                
                cPickle.dump(cell.id,pf)       
                cPickle.dump(dictToPickle,pf)       
                
            except LookupError,e:
                pass
            

        pf.close()
        self.appendXMLStub(_rstXMLElem,sd)
        
    
    def outputAdhesionFlexPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #AdhesionFlexPlugin 
        adhesionFlexPlugin=None
        if self.sim.pluginManager.isLoaded("AdhesionFlex"):
            import CompuCell            
            adhesionFlexPlugin=CompuCell.getAdhesionFlexPlugin()
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='AdhesionFlex'
        sd.moduleType='Plugin'
        sd.objectName='AdhesionFlex'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'AdhesionFlex'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        #wtiting medium adhesion vector
        
        mediumAdhesionVector=adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
        cPickle.dump(mediumAdhesionVector,pf)
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cellAdhesionVector=adhesionFlexPlugin.getAdhesionMoleculeDensityVector(cell)
            cPickle.dump(cellAdhesionVector,pf)        
        
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)
        
        
    def outputChemotaxisPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #ChemotaxisPlugin 
        chemotaxisPlugin=None
        if self.sim.pluginManager.isLoaded("Chemotaxis"):
            import CompuCell            
            chemotaxisPlugin=CompuCell.getChemotaxisPlugin()
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Chemotaxis'
        sd.moduleType='Plugin'
        sd.objectName='Chemotaxis'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'Chemotaxis'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:        
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        for cell in cellList:            
            cPickle.dump(cell.id,pf)     
            
            fieldNames=chemotaxisPlugin.getFileNamesWithChemotaxisData(cell)
            #outputting numbed of chemotaxis data that cell has
            cPickle.dump(len(fieldNames),pf)      
            
            for fieldName in fieldNames:                
                chd=chemotaxisPlugin.getChemotaxisData(cell,fieldName)
                chdDict={}
                chdDict['fieldName']=fieldName
                chdDict['lambda']=chd.getLambda()
                chdDict['saturationCoef']=chd.saturationCoef
                chdDict['formulaName']=chd.formulaName
                chemotactTowardsVec=chd.getChemotactTowardsVectorTypes()
                print 'chemotactTowardsVec=',chemotactTowardsVec
                chdDict['chemotactTowardsTypesVec']=chd.getChemotactTowardsVectorTypes()
                
                cPickle.dump(chdDict,pf)                      
            print 'fieldNames=',fieldNames
            # cPickle.dump(cellAdhesionVector,pf)        
        
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)
        
        
    def outputLengthConstraintPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #LengthConstraintPlugin 
        lengthConstraintPlugin=None        
        if self.sim.pluginManager.isLoaded("LengthConstraint"):
            import CompuCell            
            lengthConstraintPlugin=CompuCell.getLengthConstraintPlugin()            
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='LengthConstraint'
        sd.moduleType='Plugin'
        sd.objectName='LengthConstraint'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'LengthConstraint'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        lcp=lengthConstraintPlugin
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump([lcp.getLambdaLength(cell),lcp.getTargetLength(cell),lcp.getMinorTargetLength(cell)],pf)
            
        
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)

    def outputConnectivityGlobalPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #ConnectivityLocalFlexPlugin    
        connectivityGlobalPlugin=None        
        if self.sim.pluginManager.isLoaded("ConnectivityGlobal"):
            import CompuCell            
            connectivityGlobalPlugin=CompuCell.getConnectivityGlobalPlugin()            
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='ConnectivityGlobal'
        sd.moduleType='Plugin'
        sd.objectName='ConnectivityGlobal'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'ConnectivityGlobal'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
                
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump(connectivityGlobalPlugin.getConnectivityStrength(cell),pf)
            
        
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)

    def outputConnectivityLocalFlexPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #ConnectivityLocalFlexPlugin    
        connectivityLocalFlexPlugin=None        
        if self.sim.pluginManager.isLoaded("ConnectivityLocalFlex"):
            import CompuCell            
            connectivityLocalFlexPlugin=CompuCell.getConnectivityLocalFlexPlugin()            
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='ConnectivityLocalFlex'
        sd.moduleType='Plugin'
        sd.objectName='ConnectivityLocalFlex'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'ConnectivityLocalFlex'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
                
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump(connectivityLocalFlexPlugin.getConnectivityStrength(cell),pf)
            
        
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)

    def outputFocalPointPlacticityPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #FocalPointPlasticity
        focalPointPlasticityPlugin=None
        if self.sim.pluginManager.isLoaded("FocalPointPlasticity"):
            import CompuCell
            focalPointPlasticityPlugin=CompuCell.getFocalPointPlasticityPlugin()         
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='FocalPointPlasticity'
        sd.moduleType='Plugin'
        sd.objectName='FocalPointPlasticity'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'FocalPointPlasticity'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in cellList:            
        
            cPickle.dump(cell.id,pf)            
            fppVec=focalPointPlasticityPlugin.getFPPDataVec(cell)
            internalFPPVec=focalPointPlasticityPlugin.getInternalFPPDataVec(cell)
            anchorFPPVec=focalPointPlasticityPlugin.getAnchorFPPDataVec(cell)
            
            # dumping 'external' fpp links
            cPickle.dump(len(fppVec),pf)           
            for fppData in fppVec:
                fppDataDict={}
                if fppData.neighborAddress:                    
                    fppDataDict['neighborIds']=[fppData.neighborAddress.id,fppData.neighborAddress.clusterId]
                else:
                    fppDataDict['neighborIds']=[0,0]
                fppDataDict['lambdaDistance']=fppData.lambdaDistance
                fppDataDict['targetDistance']=fppData.targetDistance
                fppDataDict['maxDistance']=fppData.maxDistance
                fppDataDict['activationEnergy']=fppData.activationEnergy
                fppDataDict['maxNumberOfJunctions']=fppData.maxNumberOfJunctions
                fppDataDict['neighborOrder']=fppData.neighborOrder
                cPickle.dump(fppDataDict,pf)
                
            # dumping 'internal' fpp links
            cPickle.dump(len(internalFPPVec),pf)           
            for fppData in internalFPPVec:
                fppDataDict={}
                if fppData.neighborAddress:                    
                    fppDataDict['neighborIds']=[fppData.neighborAddress.id,fppData.neighborAddress.clusterId]
                else:
                    fppDataDict['neighborIds']=[0,0]
                fppDataDict['lambdaDistance']=fppData.lambdaDistance
                fppDataDict['targetDistance']=fppData.targetDistance
                fppDataDict['maxDistance']=fppData.maxDistance
                fppDataDict['activationEnergy']=fppData.activationEnergy
                fppDataDict['maxNumberOfJunctions']=fppData.maxNumberOfJunctions
                fppDataDict['neighborOrder']=fppData.neighborOrder
                cPickle.dump(fppDataDict,pf)


            # dumping anchor fpp links
            cPickle.dump(len(anchorFPPVec),pf)           
            for fppData in anchorFPPVec:
                fppDataDict={}
                fppDataDict['lambdaDistance']=fppData.lambdaDistance
                fppDataDict['targetDistance']=fppData.targetDistance
                fppDataDict['maxDistance']=fppData.maxDistance
                fppDataDict['anchorId']=fppData.anchorId                
                fppDataDict['anchorPoint']=[fppData.anchorPoint[0],fppData.anchorPoint[1],fppData.anchorPoint[2]]                                                
                cPickle.dump(fppDataDict,pf)

                
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)

    def outputContactLocalProductPlugin(self,_restartOutputPath,_rstXMLElem):
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #ContactLocalProductPlugin 
        contactLocalProductPlugin=None
        if self.sim.pluginManager.isLoaded("ContactLocalProduct"):
            import CompuCell            
            contactLocalProductPlugin=CompuCell.getContactLocalProductPlugin()         
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='ContactLocalProduct'
        sd.moduleType='Plugin'
        sd.objectName='ContactLocalProduct'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'ContactLocalProduct'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
                
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump(contactLocalProductPlugin.getCadherinConcentrationVec(cell),pf)
            
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)

    def outputCellOrientationPlugin(self,_restartOutputPath,_rstXMLElem):
    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #CellOrientationPlugin 
        cellOrientationPlugin=None
        if self.sim.pluginManager.isLoaded("CellOrientation"):
            import CompuCell            
            cellOrientationPlugin=CompuCell.getCellOrientationPlugin()    
        else:
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='CellOrientation'
        sd.moduleType='Plugin'
        sd.objectName='CellOrientation'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'CellOrientation'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump(cellOrientationPlugin.getLambdaCellOrientation(cell),pf)
                    
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)
        
    def outputPolarizationVectorPlugin(self,_restartOutputPath,_rstXMLElem):
    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #PolarizationVectorPlugin 
        polarizationVectorPlugin=None
        if self.sim.pluginManager.isLoaded("PolarizationVector"):
            import CompuCell            
            polarizationVectorPlugin=CompuCell.getPolarizationVectorPlugin()
        else:    
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='PolarizationVector'
        sd.moduleType='Plugin'
        sd.objectName='PolarizationVector'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'PolarizationVector'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            cPickle.dump(polarizationVectorPlugin.getPolarizationVector(cell),pf)
                    
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)


    def outputPolarization23Plugin(self,_restartOutputPath,_rstXMLElem):
    
        import SerializerDEPy
        import CompuCellSetup    
        import cPickle
        from PySteppables import CellList   
        import CompuCell
        
        #polarization23Plugin 
        polarization23Plugin=None
        if self.sim.pluginManager.isLoaded("Polarization23"):
            import CompuCell            
            polarization23Plugin=CompuCell.getPolarization23Plugin()
        else:    
            return
            
        sd=SerializerDEPy.SerializeData()
        sd.moduleName='Polarization23'
        sd.moduleType='Plugin'
        sd.objectName='Polarization23'
        sd.objectType='Pickle'        
        sd.fileName=os.path.join(_restartOutputPath,'Polarization23'+'.dat')
        
        
        
        inventory = self.sim.getPotts().getCellInventory()                
        cellList = CellList(inventory)        
        numberOfCells=len(cellList)

        try:
            pf=open(sd.fileName,'w')
        except IOError,e:
            return
        
        cPickle.dump(numberOfCells,pf)
        
        for cell in cellList:            
            cPickle.dump(cell.id,pf)            
            polVec=polarization23Plugin.getPolarizationVector(cell)            
            cPickle.dump([polVec.fX,polVec.fY,polVec.fZ],pf)
            cPickle.dump(polarization23Plugin.getPolarizationMarkers(cell),pf)
            cPickle.dump(polarization23Plugin.getLambdaPolarization(cell),pf)
                    
        pf.close()        
        self.appendXMLStub(_rstXMLElem,sd)
    