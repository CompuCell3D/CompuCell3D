# -*- coding: utf-8 -*-


# self.FIELD_TYPES = ("CellField", "ConField", "ScalarField", "ScalarFieldCellLevel" , "VectorField","VectorFieldCellLevel")
    
MODULENAME = '------- CMLFieldHandler.py: '

class CMLFieldHandler():
    
    def __init__(self):
        import PlayerPython    # swig'd from core/pyinterface/PlayerPythonNew/PlayerPython.i
        
        self.fieldStorage = PlayerPython.FieldStorage()
        self.fieldWriter = PlayerPython.FieldWriter()
        self.fieldWriter.setFieldStorage(self.fieldStorage)
#        self.fieldWriter.setFileTypeToBinary(False)  # not currently being used
        self.fieldTypes = {}
        self.outputFrequency = 1
        self.sim = None
        self.outputDirName = ""
        self.outputFileCoreName = "Step"
        self.outFileNumberOfDigits = 0;
        self.doNotOutputFieldList=[]
        self.FIELD_TYPES = ("CellField", "ConField", "ScalarField", "ScalarFieldCellLevel" , "VectorField","VectorFieldCellLevel")
        
    def doNotOutputField(self,_fieldName):
        if not _fieldName in  self.doNotOutputFieldList:
            self.doNotOutputFieldList.append(_fieldName)
        
    def setFieldStorage(self,_fieldStorage):
        self.fieldStorage = _fieldStorage
        self.fieldWriter.setFieldStorage(self.fieldStorage)
        
    def createVectorFieldPy(self,_dim,_fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldPy(_dim,_fieldName)
        
    def createVectorFieldCellLevelPy(self,_fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldCellLevelPy(_fieldName)
        
    def createFloatFieldPy(self, _dim,_fieldName):        
        import CompuCellSetup
        return CompuCellSetup.createFloatFieldPy(_dim,_fieldName)
        
    def createScalarFieldCellLevelPy(self,_fieldName):
        import CompuCellSetup
        return CompuCellSetup.createScalarFieldCellLevelPy(_fieldName)
        
    def clearGraphicsFields(self):
        pass
        
    def setMaxNumberOfSteps(self,_max):
        self.outFileNumberOfDigits = len(str(_max))    
        
    def writeFields(self,_mcs):
        from string import zfill
        from os.path import join
        
#        print MODULENAME,' writeFields():  self.fieldTypes.keys()=',self.fieldTypes.keys()
        for fieldName in self.fieldTypes.keys():
            if self.fieldTypes[fieldName]==self.FIELD_TYPES[0]:
                self.fieldWriter.addCellFieldForOutput()
            elif self.fieldTypes[fieldName]==self.FIELD_TYPES[1]:    
                self.fieldWriter.addConFieldForOutput(fieldName)
            elif self.fieldTypes[fieldName]==self.FIELD_TYPES[2]:    
                self.fieldWriter.addScalarFieldForOutput(fieldName)
            elif self.fieldTypes[fieldName]==self.FIELD_TYPES[3]:    
                self.fieldWriter.addScalarFieldCellLevelForOutput(fieldName)
            elif self.fieldTypes[fieldName]==self.FIELD_TYPES[4]:    
                self.fieldWriter.addVectorFieldForOutput(fieldName)
            elif self.fieldTypes[fieldName]==self.FIELD_TYPES[5]:    
                self.fieldWriter.addVectorFieldCellLevelForOutput(fieldName)
                
        mcsFormattedNumber = zfill(str(_mcs),self.outFileNumberOfDigits) # fills string with 0's up to self.screenshotNumberOfDigits width
        latticeDataFileName = join(self.outputDirName,self.outputFileCoreName+"_"+mcsFormattedNumber+".vtk")   # e.g. /path/Step_01.vtk
#        print MODULENAME,'  writeFields(): latticeDataFileName=',latticeDataFileName
        self.fieldWriter.writeFields(latticeDataFileName)
        self.fieldWriter.clear()
    
    def writeXMLDescriptionFile(self,_fileName=""):
        from os.path import join
        """
        This function will write XML description of the stored fields. It has to be called after 
        initialization of theCMLFieldHandler is completed
        """
        import CompuCellSetup
        latticeTypeStr = CompuCellSetup.ExtractLatticeType()
        if latticeTypeStr=="":
            latticeTypeStr = "Square"
        
        typeIdTypeNameDict = CompuCellSetup.ExtractTypeNamesAndIds()
        print "typeIdTypeNameDict",typeIdTypeNameDict
        
        from XMLUtils import ElementCC3D
        dim = self.sim.getPotts().getCellFieldG().getDim()
        numberOfSteps = self.sim.getNumSteps()
        latticeDataXMLElement=ElementCC3D("CompuCell3DLatticeData",{"Version":"1.0"})
        latticeDataXMLElement.ElementCC3D("Dimensions",{"x":str(dim.x),"y":str(dim.y),"z":str(dim.z)})
        latticeDataXMLElement.ElementCC3D("Lattice",{"Type":latticeTypeStr})
        latticeDataXMLElement.ElementCC3D("Output",{"Frequency":str(self.outputFrequency),"NumberOfSteps":str(numberOfSteps),"CoreFileName":self.outputFileCoreName,"Directory":self.outputDirName})
        #output information about cell type names and cell ids. It is necessary during generation of the PIF files from VTK output
        for typeId in typeIdTypeNameDict.keys():
            latticeDataXMLElement.ElementCC3D("CellType",{"TypeName":str(typeIdTypeNameDict[typeId]),"TypeId":str(typeId)})
            
        fieldsXMLElement=latticeDataXMLElement.ElementCC3D("Fields")
        for fieldName in self.fieldTypes.keys():
            fieldsXMLElement.ElementCC3D("Field",{"Name":fieldName,"Type":self.fieldTypes[fieldName]})
        # writing XML description to the disk
        if _fileName!="":
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(_fileName))
        else:
            latticeDataFileName = join(self.outputDirName,self.outputFileCoreName+"LDF.dml")
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(latticeDataFileName))
        
        
    def prepareSimulationStorageDir(self,_dirName):
        from os.path import exists
        from os import makedirs
#        MODULENAME = '------- CMLFieldHandler.py: '
           
        if self.outputFrequency:
            print '\n\n',MODULENAME,"prepareSimulationStorageDir: simulationData directory Name =",_dirName
            if exists(_dirName):
                print MODULENAME," yes, os.path.exists ",_dirName
                self.outputDirName = _dirName
            else:
                try:
                    makedirs(_dirName)
                    self.outputDirName = _dirName
                except:
                    self.outputFrequency = 0 # if directory cannot be created the simulation data will not be saved even if user requests it
                    print MODULENAME,"prepareSimulationStorageDir: COULD NOT MAKE DIRECTORY"
                    raise IOError
        else:
            print '\n\n',MODULENAME,"prepareSimulationStorageDir(): Lattice output frequency is invalid"
    
    def getInfoAboutFields(self):
        #there will always be cell field
        print MODULENAME,"getInfoAboutFields():  self.fieldTypes= ",self.fieldTypes
        print MODULENAME,"getInfoAboutFields():  self.FIELD_TYPES= " ,self.FIELD_TYPES
        self.fieldTypes["Cell_Field"] = self.FIELD_TYPES[0]
        
#        if not self.sim:
#            print MODULENAME, 'getInfoAboutFields: self.sim is null!!!!'
            
        # extracting information about concentration vectors
        concFieldNameVec = self.sim.getConcentrationFieldNameVector()
        for fieldName in concFieldNameVec:
            print "Got concentration field: ",fieldName
            if not fieldName in self.doNotOutputFieldList:
                self.fieldTypes[fieldName] = self.FIELD_TYPES[1]
            
        #inserting extra scalar fields managed from Python script
        scalarFieldNameVec=self.fieldStorage.getScalarFieldNameVector()
        for fieldName in scalarFieldNameVec:
            print MODULENAME,"Got this scalar field from Python: ",fieldName
            if not fieldName in self.doNotOutputFieldList:
                self.fieldTypes[fieldName] = self.FIELD_TYPES[2]
        
        #inserting extra scalar fields cell levee managed from Python script
        scalarFieldCellLevelNameVec=self.fieldStorage.getScalarFieldCellLevelNameVector()
        for fieldName in scalarFieldCellLevelNameVec:
            print MODULENAME,"Got this scalar cell level field from Python: ",fieldName
            if not fieldName in self.doNotOutputFieldList:
                self.fieldTypes[fieldName] = self.FIELD_TYPES[3]
            
        #inserting extra vector fields  managed from Python script
        vectorFieldNameVec=self.fieldStorage.getVectorFieldNameVector()
        for fieldName in vectorFieldNameVec:
            print MODULENAME,"Got this vector field from Python: ",fieldName
            if not fieldName in self.doNotOutputFieldList:
                self.fieldTypes[fieldName] = self.FIELD_TYPES[4]
            
        #inserting extra vector fields  cell level managed from Python script
        vectorFieldCellLevelNameVec=self.fieldStorage.getVectorFieldCellLevelNameVector()
        for fieldName in vectorFieldCellLevelNameVec:
            print MODULENAME,"Got this vector cell level field from Python: ",fieldName
            if not fieldName in self.doNotOutputFieldList:
                self.fieldTypes[fieldName] = self.FIELD_TYPES[5]
            
            