import CompuCell
from PySteppables import *

class VolumeConstraintSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        
    def start(self):
        for cell in self.cellList:
            cell.targetVolume=25
            cell.lambdaVolume=2.0
    def step(self,mcs):
        field=CompuCell.getConcentrationField(self.simulator,"FGF")
        comPt=CompuCell.Point3D()
        for cell in self.cellList:
            if cell.type==1: #Condensing cell
                comPt.x=int(round(cell.xCM/float(cell.volume)))
                comPt.y=int(round(cell.yCM/float(cell.volume)))
                comPt.z=int(round(cell.zCM/float(cell.volume)))
                concentration=field.get(comPt) # get concentration at comPt
                cell.targetVolume+=0.1*concentration # increase cell's target volume

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
from PyPluginsExamples import MitosisPyPluginBase
class CellsortMitosis(MitosisPyPluginBase):
    def __init__(self , _simulator , _changeWatcherRegistry , _stepperRegistry):
        MitosisPyPluginBase.__init__(self,_simulator,_changeWatcherRegistry, _stepperRegistry)
    def updateAttributes(self):
        self.parentCell.targetVolume/=2.0
        self.childCell.targetVolume=self.parentCell.targetVolume
        self.childCell.lambdaVolume=self.parentCell.lambdaVolume

        if (random()<0.5):
            self.childCell.type=self.parentCell.type
        else:
            self.childCell.type=3

        #get a reference to lists storing Mitosis data
        parentCellList=CompuCell.getPyAttrib(self.parentCell)
        childCellList=CompuCell.getPyAttrib(self.childCell)

        ##will record mitosis data in parent and offspring cells
        mcs=self.simulator.getStep()
        mitData=MitosisData(mcs,self.parentCell.id,self.parentCell.type,self.childCell.id,self.childCell.type)
        parentCellList.append(mitData)
        childCellList.append(mitData)
        

class MitosisDataPrinterSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        
    def step(self,mcs):
        for cell in self.cellList:
            mitDataList=CompuCell.getPyAttrib(cell)
            if len(mitDataList) > 0:
                print "MITOSIS DATA FOR CELL ID",cell.id
                for mitData in mitDataList:
                    print mitData
