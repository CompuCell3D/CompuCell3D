import re
import string

class SteppableTemplates:
    def __init__(self):
        self.steppableTemplatesDict={}
        
        self.initSteppableTemplates()
        
        
    def getSteppableTemplatesDict(self):
        return self.steppableTemplatesDict

        
    def generateSteppableCode(self,_steppableName="GenericSteppable",_frequency=1,_type="Generic",_extraFields=[]):
    
        try:
            text=self.steppableTemplatesDict[_type]
        except LookupError,e:
            return ""

        
        text=re.sub("STEPPABLENAME",_steppableName,text)        
        text=re.sub("FREQUENCY",str(_frequency),text)
        
        extraFieldsCode=''
        if "Scalar" in _extraFields:
            extraFieldsCode+="""
        self.scalarField=CompuCellSetup.createScalarFieldPy(self.dim,"FIELD_NAME_S")    
"""
        if "ScalarCellLevel" in _extraFields:
            extraFieldsCode+="""
        self.scalarCLField=CompuCellSetup.createScalarFieldCellLevelPy("FIELD_NAME_SCL")
"""
        if "Vector" in _extraFields:
            extraFieldsCode+="""
        self.vectorField=CompuCellSetup.createVectorFieldPy(self.dim,"FIELD_NAME_V")
"""
        if "VectorCellLevel" in _extraFields:
            extraFieldsCode+="""
        self.vectorCLField=CompuCellSetup.createVectorFieldCellLevelPy("FIELD_NAME_VCL")
"""
        text=re.sub("EXTRAFIELDS",extraFieldsCode,text)

        return text
        
    def  generateSteppableRegistrationCode(self,_steppableName="GenericSteppable",_frequency=1,_steppableFile="",_indentationLevel=0,_indentationWidth=4):
        try:
            text=self.steppableTemplatesDict["SteppableRegistrationCode"]
        except LookupError,e:
            return ""
        
        text=re.sub("STEPPABLENAME",_steppableName,text)        
        text=re.sub("STEPPABLEFILE",_steppableFile,text)                
        text=re.sub("FREQUENCY",str(_frequency),text)
        
        # possible indentation of registration code - quite unlikely it wiil be needed
        if _indentationLevel<0:
            _indentationLevel=0
        
        textLines=text.splitlines(True)
        
        for i in range(len(textLines)):
            textLines[i]=' '*_indentationWidth*_indentationLevel+textLines[i]
        
        text=''.join(textLines)
        
        return text
        
    
    def initSteppableTemplates(self):
        self.steppableTemplatesDict["SteppableRegistrationCode"]="""
from STEPPABLEFILE import STEPPABLENAME
instanceOfSTEPPABLENAME=STEPPABLENAME(_simulator=sim,_frequency=FREQUENCY)
steppableRegistry.registerSteppable(instanceOfSTEPPABLENAME)

"""    
    
        self.steppableTemplatesDict["Generic"]="""
from PySteppables import *
import CompuCell
import sys

from PlayerPython import *
import CompuCellSetup
from math import *


class STEPPABLENAME(SteppableBasePy):
    def __init__(self,_simulator,_frequency=FREQUENCY):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        EXTRAFIELDS
    def start(self):
        print "STEPPABLENAME: This function is called once before simulation"
        
    def step(self,mcs):
        print "STEPPABLENAME: This function is called every FREQUENCY MCS"
        for cell in self.cellList:
            print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
            
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return
    
"""        

        self.steppableTemplatesDict["RunBeforeMCS"]="""
from PySteppables import *
import CompuCell
import sys

from PlayerPython import *
from math import *

class STEPPABLENAME(RunBeforeMCSSteppableBasePy):
    def __init__(self,_simulator,_frequency=FREQUENCY):
        RunBeforeMCSSteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
        print "STEPPABLENAME: This function is called once before simulation"
        
        
    def step(self,mcs):
        print "STEPPABLENAME: This function is called before MCS i.e. pixel-copies take place for that MCS "    
        print "STEPPABLENAME: This function is called every FREQUENCY MCS "
        
        # typical use for this type of steppable is secretion - uncomment lines  below and include Secretion plugin to make commented code work
        # attrSecretor=self.getFieldSecretor("FIELD TO SECRETE")
        # for cell in self.cellList:
            # if cell.type==3:
                # attrSecretor.secreteInsideCell(cell,300)
                # attrSecretor.secreteInsideCellAtBoundary(cell,300)
                # attrSecretor.secreteOutsideCellAtBoundary(cell,500)
                # attrSecretor.secreteInsideCellAtCOM(cell,300)             
            
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return
    
"""        



        self.steppableTemplatesDict["Mitosis"]="""
from PySteppables import *
from PySteppablesExamples import MitosisSteppableBase
import CompuCell
import sys

from PlayerPython import *
from math import *


class STEPPABLENAME(MitosisSteppableBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
    
    def step(self,mcs):
        # print "INSIDE STEPPABLENAME"
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
        
"""        

        self.steppableTemplatesDict["ClusterMitosis"]="""
from PySteppables import *
from PySteppablesExamples import MitosisSteppableClustersBase
import CompuCell
import sys

from PlayerPython import *
from math import *


class STEPPABLENAME(MitosisSteppableClustersBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableClustersBase.__init__(self,_simulator, _frequency)           
        
        
    def step(self,mcs):        
        
        # print "INSIDE STEPPABLENAME"
        
        for cell in self.cellList:            
            clusterCellList=self.getClusterCells(cell.clusterId)
            print "DISPLAYING CELL IDS OF CLUSTER ",cell.clusterId,"CELL. ID=",cell.id
            for cellLocal in clusterCellList:
                print "CLUSTER CELL ID=",cellLocal.id," type=",cellLocal.type
                
                
        
        mitosisClusterIdList=[]
        for compartmentList in self.clusterList:
            # print "cluster has size=",compartmentList.size()
            clusterId=0
            clusterVolume=0            
            for cell in CompartmentList(compartmentList):
                clusterVolume+=cell.volume            
                clusterId=cell.clusterId
            
            
            if clusterVolume>250: # condition under which cluster mitosis takes place
                mitosisClusterIdList.append(clusterId) # instead of doing mitosis right away we store ids for clusters which should be divide. This avoids modifying cluster list while we iterate through it
        for clusterId in mitosisClusterIdList:
            # to change mitosis mode leave one of the below lines uncommented
            
            # self.divideClusterOrientationVectorBased(clusterId,1,0,0)             # this is a valid option
            self.divideClusterRandomOrientation(clusterId)
            # self.divideClusterAlongMajorAxis(clusterId)                                # this is a valid option
            # self.divideClusterAlongMinorAxis(clusterId)                                # this is a valid option
            

    def updateAttributes(self):
        # compartments in the parent and child clusters arel listed in the same order so attribute changes require simple iteration through compartment list  
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
                
        compartmentListChild=self.inventory.getClusterCells(childCell.clusterId)
        compartmentListParent=self.inventory.getClusterCells(parentCell.clusterId)
        print "compartmentListChild=",compartmentListChild 
        for i in xrange(compartmentListChild.size()):
            compartmentListParent[i].targetVolume/=2.0
            # compartmentListParent[i].targetVolume=25
            compartmentListChild[i].targetVolume=compartmentListParent[i].targetVolume
            compartmentListChild[i].lambdaVolume=compartmentListParent[i].lambdaVolume

        
"""        

