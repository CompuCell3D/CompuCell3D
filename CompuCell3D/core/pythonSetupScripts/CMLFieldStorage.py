# -*- coding: utf-8 -*-
import os,sys


FIELD_TYPES = ("CellField", "ConField", "ScalarField", "ScalarFieldCellLevel" , "VectorField","VectorFieldCellLevel")
    
class CMLFieldHandler():

    def __init__(self):
        import PlayerPython
        self.fieldStorage=PlayerPython.FieldStorage()
        self.fieldWriter=PlayerPython.FieldWriter()
        self.fieldTypes={}
        self.outputFrequency=1
        self.sim=None
        self.outputDirName=""
        self.outputFileCoreName=""
        self.outFileNumberOfDigits=0;
        
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
        self.outFileNumberOfDigits=len(str(_max))    
        
    def writeFields(self,_mcs):
        import os.path
        import string
        for fieldName in self.fieldTypes.keys():
            if self.fieldTypes[fieldName]==FIELD_TYPES[0]:
                self.fieldWriter.addCellFieldForOutput()
            elif self.fieldTypes[fieldName]==FIELD_TYPES[1]:    
                self.fieldWriter.addConFieldForOutput(fieldName)
        mcsFormattedNumber = string.zfill(str(_mcs),self.outFileNumberOfDigits) # fills string wtih 0's up to self.screenshotNumberOfDigits width
        latticeDataFileName=os.path.join(self.outputDirName,self.outputFileCoreName+"_"+mcsFormattedNumber+".vtk")
        self.fieldWriter.writeFields(latticeDataFileName)
        self.fieldWriter.clear()
    
    def writeXMLDescriptionFile(self,_fileName=""):
        """
        This function will write XML description of the stored fields. It has to be called after 
        initializetion of theCMLFieldHandler is completed
        """

        
        from XMLUtils import ElementCC3D
        dim=self.sim.getPotts().getCellFieldG().getDim()        
        latticeDataXMLElement=ElementCC3D("CompuCell3DLatticeData",{"Version":"1.0"})
        latticeDataXMLElement.ElementCC3D("Dimensions",{"x":str(dim.x),"y":str(dim.y),"z":str(dim.z)})
        latticeDataXMLElement.ElementCC3D("Output",{"Frequency":str(self.outputFrequency),"CoreFileName":self.outputFileCoreName,"Directory":self.outputDirName})
        
        fieldsXMLElement=latticeDataXMLElement.ElementCC3D("Fields")
        for fieldName in self.fieldTypes.keys():
            fieldsXMLElement.ElementCC3D("Field",{"Name":fieldName,"Type":self.fieldTypes[fieldName]})
        # writing XML description to the disk
        if _fileName!="":
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(_fileName))
        else:
            latticeDataFileName=os.path.join(self.outputDirName,self.outputFileCoreName+"LDF.dml")
            latticeDataXMLElement.CC3DXMLElement.saveXML(str(latticeDataFileName))
        
        
    def prepareSimulationStorageDir(self,_dirName):
        if self.outputFrequency:
            print "simulationData directory Name =",_dirName
            if not os.path.isdir(_dirName):
                os.mkdir(_dirName)
                self.outputDirName=_dirName
            else:
                self.outputFrequency=0 # if directory cannot be created the simulation data will not be saved even if user requests it
    
    def getInfoAbutFields(self):
        #there will always be cell field
        self.fieldTypes["Cell_Field"]=FIELD_TYPES[0]
        
        # extracting information about concentration vectors
        concFieldNameVec=self.sim.getConcentrationFieldNameVector()
        for fieldName in concFieldNameVec:
            print "Got concentration field: ",fieldName
            self.fieldTypes[fieldName]=FIELD_TYPES[1]
            
        #inserting extra scalar fields managed from Python script
        scalarFieldNameVec=self.fieldStorage.getScalarFieldNameVector()
        for fieldName in scalarFieldNameVec:
            print "Got this scalar field from Python: ",fieldName
            self.fieldTypes[fieldName]=FIELD_TYPES[2]
        
        #inserting extra scalar fields cell levee managed from Python script
        scalarFieldCellLevelNameVec=self.fieldStorage.getScalarFieldCellLevelNameVector()
        for fieldName in scalarFieldCellLevelNameVec:
            print "Got this scalar cell level field from Python: ",fieldName
            self.fieldTypes[fieldName]=FIELD_TYPES[3]
            
        #inserting extra vector fields  managed from Python script
        vectorFieldNameVec=self.fieldStorage.getVectorFieldNameVector()
        for fieldName in vectorFieldNameVec:
            print "Got this vector field from Python: ",fieldName
            self.fieldTypes[fieldName]=FIELD_TYPES[4]
            
        #inserting extra vector fields  cell level managed from Python script
        vectorFieldCellLevelNameVec=self.fieldStorage.getVectorFieldCellLevelNameVector()
        for fieldName in vectorFieldCellLevelNameVec:
            print "Got this vector cell level field from Python: ",fieldName
            self.fieldTypes[fieldName]=FIELD_TYPES[5]
            
            