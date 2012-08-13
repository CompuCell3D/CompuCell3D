import os.path
import sys
import string

def generateConfigureSimulationHeader():
    configureSimLines='''
def configureSimulation(sim):
    import CompuCellSetup
    from XMLUtils import ElementCC3D
    
'''
    return configureSimLines

def generateConfigureSimFcnBody(_rootElement,_outputFileName):
    # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt - here it will be CompuCell3DElmnt
    configureSimFileName=str(_outputFileName)
    
    _rootElement.saveXMLInPython(configureSimFileName)         
    
    configureSimLines=generateConfigureSimulationHeader()        
    
    configureSimFile=open(configureSimFileName,"r")
    configureSimBody=configureSimFile.read()
    configureSimFile.close()    
    configureSimLines+=configureSimBody
    configureSimLines+='''
    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)    
    '''
    configureSimLines+='\n'
    
    
    os.remove(configureSimFileName)
    
    return configureSimLines


class CC3DPythonGenerator:
    def __init__(self,_xmlGenerator):
        self.xmlGenerator=_xmlGenerator
        self.simulationDir=self.xmlGenerator.simulationDir
        self.simulationName=self.xmlGenerator.simulationName
        self.xmlFileName=self.xmlGenerator.fileName
        
        self.mainPythonFileName=os.path.join(str(self.simulationDir),str(self.simulationName)+".py")
        self.steppablesPythonFileName=os.path.join(str(self.simulationDir),str(self.simulationName)+"Steppables.py")        
        
        self.configureSimLines=''
        
        self.attachDictionary=False
        self.attachList=False
        self.plotTypeTable=[]
        self.pythonPlotsLines=''
        
        self.pythonPlotsNames=[]        
        
        self.steppableCodeLines=''
        self.steppableRegistrationLines=''
        
        self.generatedSteppableNames=[]
        self.generatedVisPlotSteppableNames=[]
        
        self.cellTypeTable=[["Medium",False]]
        self.afMolecules=[]
        self.afFormula='min(Molecule1,Molecule2)'
        self.cmcCadherins=[]                
        
        self.pythonOnlyFlag=False
        
        self.steppableFrequency=1
        
    def setPythonOnlyFlag(self,_flag):
        self.pythonOnlyFlag=_flag
    
    # def setCMCTable(self,_table):
        # self.cmcCadherins=_table

    # def setAFFormula(self,_formula):        
        # self.afFormula=_formula
        
    # def setAFTable(self,_table):
        # self.afMolecules=_table
        
    # def setCellTypeTable(self,_table):
        # self.cellTypeTable=_table
        # #generate typeId to typeTuple lookup dictionary
        
        # self.idToTypeTupleDict={}
        # typeCounter=0
        
        # for typeTupple in self.cellTypeTable:
            # self.idToTypeTupleDict[typeCounter]=typeTupple            
            # typeCounter+=1
            
    def setPlotTypeTable(self,_table):
        self.plotTypeTable=_table
        
        if not len(self.plotTypeTable):
            return
        
        self.pythonPlotsLines='''
# -------------- extra fields  -------------------      
dim=sim.getPotts().getCellFieldG().getDim()        
        '''
        for plotTupple in self.plotTypeTable:
            plotName=plotTupple[0]
                
            plotType=plotTupple[1]
            if plotType=="ScalarField":
                fieldLines='''
%sVisField=simthread.createFloatFieldPy(dim,"%s")                
                ''' %(plotName,plotName)
                self.pythonPlotsLines+=fieldLines
                self.pythonPlotsNames.append((('%sVisField')%plotName,'ScalarField'))                
                
            elif plotType=="CellLevelScalarField":
                fieldLines='''
%sVisField=simthread.createScalarFieldCellLevelPy("%s")                
                ''' %(plotName,plotName)
                self.pythonPlotsLines+=fieldLines
                self.pythonPlotsNames.append((('%sVisField')%plotName,'CellLevelScalarField'))                
                
            elif plotType=="VectorField":
                fieldLines='''
%sVisField=simthread.createVectorFieldPy(dim,"%s") 
                ''' %(plotName,plotName)
                self.pythonPlotsLines+=fieldLines
                self.pythonPlotsNames.append((('%sVisField')%plotName,'VectorField'))                
                
            elif plotType=="CellLevelVectorField":
                fieldLines='''
%sVisField=simthread.createVectorFieldCellLevelPy("%s") 
                ''' %(plotName,plotName)
                self.pythonPlotsLines+=fieldLines
                self.pythonPlotsNames.append((('%sVisField')%plotName,'CellLevelVectorField'))                
                
        self.pythonPlotsLines+='''
# --------------end of extra fields  -------------------      

        '''
        
    def generateConfigureSimFcn(self):
        # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt - here it will be CompuCell3DElmnt
        configureSimFileName=str(self.xmlFileName+".py")
        
        self.configureSimLines=generateConfigureSimFcnBody(self.xmlGenerator.cc3d.CC3DXMLElement,configureSimFileName)
        self.configureSimLines+='\n'
        
        # self.xmlGenerator.cc3d.CC3DXMLElement.saveXMLInPython(configureSimFileName)         
        
        # self.generateConfigureSimulationHeader()        
        
        # configureSimFile=open(configureSimFileName,"r")
        # configureSimBody=configureSimFile.read()
        # configureSimFile.close()
        # self.configureSimLines+=configureSimBody
        # os.remove(configureSimFileName)

        
    def generateMainPythonScript(self):
        file=open(self.mainPythonFileName,"w")
        print "self.pythonPlotsLines=",self.pythonPlotsLines
        header=''
        
        if  self.pythonOnlyFlag:
            self.generateConfigureSimFcn()
            
        # note the XML root element generated using C++ xml-to-python converter is called RootElementNameElmnt - here it will be CompuCell3DElmnt        
        if self.configureSimLines!='':
            header+=self.configureSimLines
            header+='''
            
    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)
            '''
            
            
            
        header+='''
import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()
        '''
        if self.configureSimLines!='':
            header+='''
configureSimulation(sim)            
            '''
        header+='''    
# add extra attributes here
        '''
        attachListLine='''
pyAttributeListAdder,listAdder=CompuCellSetup.attachListToCells(sim)
        '''
        attachDicttionaryLine='''
pyAttributeDictionaryAdder,dictAdder=CompuCellSetup.attachDictionaryToCells(sim)
        '''
        
        initSimObjectLine='''    
CompuCellSetup.initializeSimulationObjects(sim,simthread)
# Definitions of additional Python-managed fields go here
        '''
        steppableRegistryLine='''
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()
        '''
        # steppableRegistrationLine='''
# from %s import %s
# steppableInstance=%s(sim,_frequency=100)
# steppableRegistry.registerSteppable(steppableInstance)
        # '''%(self.simulationName+"Steppables",self.simulationName+"Steppable",self.simulationName+"Steppable" )
        
        mainLoopLine='''
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
        
        '''
        script=header
        if self.attachDictionary:
            script+=attachDicttionaryLine
        if self.attachList:
            script+=attachListLine
            
        script+=initSimObjectLine
        
        if self.pythonPlotsLines!='':
            script+=self.pythonPlotsLines
            
        script+=steppableRegistryLine
        script+=self.steppableRegistrationLines
        script+=mainLoopLine
        
        file.write(script)
        file.close()
        
    def generateSteppableRegistrationLines(self):
        if not len(self.generatedSteppableNames) and not len(self.generatedVisPlotSteppableNames): #using only demo steppable
        
            self.steppableRegistrationLines+='''
from %s import %s
steppableInstance=%s(sim,_frequency=%s)
steppableRegistry.registerSteppable(steppableInstance)
        '''%(self.simulationName+"Steppables",self.simulationName+"Steppable",self.simulationName+"Steppable", self.steppableFrequency )
        else:# generating registration lines for user stppables
            for steppableName in self.generatedSteppableNames:
                self.steppableRegistrationLines+='''

from %s import %s
%s=%s(sim,_frequency=%s)
steppableRegistry.registerSteppable(%s)
        '''%(self.simulationName+"Steppables",steppableName,steppableName+"Instance",steppableName, self.steppableFrequency, steppableName+"Instance")
                

    def generatePlotSteppableRegistrationLines(self):
            for plotNameTuple in self.pythonPlotsNames:
                steppableName=plotNameTuple[0]+'Steppable'
                steppableInstanceName=steppableName+"Instance"
                self.steppableRegistrationLines+='''

from %s import %s
%s=%s(sim,_frequency=%s)
%s.visField=%s
steppableRegistry.registerSteppable(%s)

        '''%(self.simulationName+"Steppables",steppableName,steppableInstanceName,steppableName, self.steppableFrequency, steppableInstanceName,plotNameTuple[0],steppableInstanceName)
        
        
    def generateVisPlotSteppables(self):
            if not len(self.pythonPlotsNames):
                return
                
            self.steppableCodeLines+='''
            
from PlayerPython import *
from math import *            
'''            
            for plotNameTuple in self.pythonPlotsNames:
                steppableName=plotNameTuple[0]+'Steppable'
                if steppableName not in self.generatedVisPlotSteppableNames:
                    self.generatedVisPlotSteppableNames.append(steppableName)
                
                plotType=plotNameTuple[1]                
                
                if plotType=='ScalarField':
                    self.steppableCodeLines+='''

class %s(SteppableBasePy):
'''%(steppableName)
                    self.steppableCodeLines+='''
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.visField=None
        
    def step(self,mcs):
        clearScalarField(self.dim,self.visField)
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    if (not mcs % 20):
                        value=x*y
                        fillScalarValue(self.visField,x,y,z,value) # value assigned to individual pixel
                    else:
                        value=sin(x*y)
                        fillScalarValue(self.visField,x,y,z,value) # value assigned to individual pixel                    
'''%(self.steppableFrequency)
                
                elif plotType=='CellLevelScalarField':
                    self.steppableCodeLines+='''                
                    
class %s(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.visField=None

    def step(self,mcs):
        clearScalarValueCellLevel(self.visField)
        from random import random
        for cell in self.cellList:
            fillScalarValueCellLevel(self.visField,cell,cell.id*random())   # value assigned to every cell , all cell pixels are painted based on this value             
'''%(steppableName,self.steppableFrequency)

                elif plotType=='VectorField':
                    self.steppableCodeLines+='''            
                    
class %s(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.visField=None
    
    def step(self,mcs):
        maxLength=0
        clearVectorField(self.dim,self.visField)        
        for x in xrange(0,self.dim.x,5):
            for y in xrange(0,self.dim.y,5):
                for z in xrange(self.dim.z):                     
                    pt=CompuCell.Point3D(x,y,z)                    
                    insertVectorIntoVectorField(self.visField,pt.x, pt.y,pt.z, pt.x, pt.y, pt.z) # vector assigned to individual pixel
'''%(steppableName,self.steppableFrequency)

                elif plotType=='CellLevelVectorField':
                    self.steppableCodeLines+='''                
                    
class %s(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.visField=None

    def step(self,mcs):
        clearVectorCellLevelField(self.visField)
        for cell in self.cellList:
            if cell.type==1:
                insertVectorIntoVectorCellLevelField(self.visField,cell, cell.id, cell.id, 0.0)
'''%(steppableName,self.steppableFrequency)

    
    def generateConstraintInitializer(self):
        if "ConstraintInitializerSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("ConstraintInitializerSteppable")
            self.steppableCodeLines+='''

class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        for cell in self.cellList:
            cell.targetVolume=25
            cell.lambdaVolume=2.0
        
        '''%(self.steppableFrequency)
            
    def generateGrowthSteppable(self):
        self.generateConstraintInitializer()
        if "GrowthSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("GrowthSteppable")
            
            self.steppableCodeLines+='''

class GrowthSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def step(self,mcs):
        for cell in self.cellList:
            cell.targetVolume+=1        
    # alternatively if you want to make growth a function of chemical concentration uncomment lines below and comment lines above        
        # field=CompuCell.getConcentrationField(self.simulator,"PUT_NAME_OF_CHEMICAL_FIELD_HERE")
        # pt=CompuCell.Point3D()
        # for cell in self.cellList:
            # pt.x=int(cell.xCOM)
            # pt.y=int(cell.yCOM)
            # pt.z=int(cell.zCOM)
            # concentrationAtCOM=field.get(pt)
            # cell.targetVolume+=0.01*concentrationAtCOM  # you can use here any fcn of concentrationAtCOM     
        
        '''%(self.steppableFrequency)
    def generateMitosisSteppable(self):
        self.generateGrowthSteppable()
        if "MitosisSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("MitosisSteppable")
            
            self.steppableCodeLines+='''

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,_simulator,_frequency=%s):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
    
    def step(self,mcs):
        # print "INSIDE MITOSIS STEPPABLE"
        cells_to_divide=[]
        for cell in self.cellList:
            if cell.volume>50:
                
                cells_to_divide.append(cell)
                
        for cell in cells_to_divide:
            # to change mitosis mode leave one of the below lines uncommented
            self.divideCellRandomOrientation(cell)
            # self.divideCellOrientationVectorBased(cell,1,0,0)                 # this is a valid option
            # self.divideCellAlongMajorAxis(cell)                               # this is a valid option
            # self.divideCellAlongMinorAxis(cell)                               # this is a valid option

    def updateAttributes(self):
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
        
        childCell.targetVolume=parentCell.targetVolume
        childCell.lambdaVolume=parentCell.lambdaVolume
        if parentCell.type==1:
            childCell.type=2
        else:
            childCell.type=1
        
        '''%(self.steppableFrequency)        
    def generateDeathSteppable(self):
        self.generateConstraintInitializer()
        if "DeathSteppable" not in self.generatedSteppableNames:
            self.generatedSteppableNames.append("DeathSteppable")
            
            self.steppableCodeLines+='''

class DeathSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def step(self,mcs):
        if mcs==1000:
            for cell in self.cellList:
                if cell.type==1:
                    cell.targetVolume==0
                    cell.lambdaVolume==100
        
        '''%(self.steppableFrequency)        
    def generateSteppablePythonScript(self):
        file=open(self.steppablesPythonFileName,"w")
        
        header='''
from PySteppables import *
import CompuCell
import sys
'''
        if "MitosisSteppable" in self.generatedSteppableNames:
            header+='''
from PySteppablesExamples import MitosisSteppableBase
            '''
            
        file.write(header)
        
        if self.steppableCodeLines=='': # writing simple demo steppable
            classDefinitionLine='''class %s(SteppableBasePy):'''%(self.simulationName+"Steppable")
            steppableBody='''    

    def __init__(self,_simulator,_frequency=%s):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        # any code in the start function runs before MCS=0
        pass
    def step(self,mcs):        
        #type here the code that will run every _frequency MCS
        for cell in self.cellList:
            print "cell.id=",cell.id
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        '''%(self.steppableFrequency)
        
            file.write(classDefinitionLine)
            file.write(steppableBody)
            
        else: # writing steppab;les according to user requests
            file.write(self.steppableCodeLines)
        file.close()    
        
            
        
        
  