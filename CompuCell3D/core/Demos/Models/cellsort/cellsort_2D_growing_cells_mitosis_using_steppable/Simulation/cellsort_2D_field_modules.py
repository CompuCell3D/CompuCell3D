import CompuCell
from PySteppables import *

class VolumeConstraintSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        for cell in self.cellList:
            cell.targetVolume=25
            cell.lambdaVolume=2.0
            cell.dict['lineage_list'] = []
            
    def step(self,mcs):
        field=CompuCell.getConcentrationField(self.simulator,"FGF")
        for cell in self.cellList:
            if cell.type==self.CONDENSING and mcs < 1500: #Condensing cell               
                concentration=field[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                cell.targetVolume+=0.1*concentration # increase cell's target volume

            if  mcs > 1500: #removing all cells
                cell.targetVolume-=1 # increase cell's target volume
                if cell.targetVolume<0.0:
                    cell.targetVolume=0.0

#MItosis data has to have base class "object" otherwise if cell will be deleted CC3D may crash due to improper garbage collection
class MitosisData(object):
    def __init__(self, _MCS=-1, _parentId=-1, _parentType=-1, _offspringId=-1, _offspringType=-1):
        self.MCS=_MCS
        self.parentId=_parentId
        self.parentType=_parentType
        self.offspringId=_offspringId
        self.offspringType=_offspringType
    def __str__(self):
        return "Mitosis time="+str(self.MCS)+" parentId="+str(self.parentId)+" offspringId="+str(self.offspringId)
        
   
from random import random
from PySteppablesExamples import MitosisSteppableBase
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
    
    def step(self,mcs):        
        cells_to_divide=[]
        for cell in self.cellList:
            if cell.volume>50:
                cells_to_divide.append(cell)   
                
        for cell in cells_to_divide:
            self.divideCellRandomOrientation(cell)                  

    def updateAttributes(self):
        self.parentCell.targetVolume /= 2.0 # reducing parent target volume                 
        self.cloneParent2Child()            

        if (random()<0.5):                    
            self.childCell.type = self.CONDENSINGDIFFERENTIATED
                        
        #will record mitosis data in parent and offspring cells
        mcs=self.simulator.getStep()
        mitData=MitosisData(mcs, self.parentCell.id, self.parentCell.type, self.childCell.id, self.childCell.type)
        self.parentCell.dict['lineage_list'].append(mitData)
        self.childCell.dict['lineage_list'].append(mitData)
        

class MitosisDataPrinterSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):
        for cell in self.cellList:
            mitDataList=cell.dict
            if len(mitDataList) > 0:
                print "MITOSIS DATA FOR CELL ID",cell.id
                for mitData in mitDataList:
                    print mitData
