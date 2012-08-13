from PySteppables import *
from PySteppablesExamples import MitosisSteppableBase
import CompuCell
import sys
from random import uniform
import math

class VolumeParamSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=1,):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin()
        self.fieldNameVEGF2 = 'VEGF2'
        self.fieldNameGlucose = 'Glucose'
                
    def start(self):
        for cell in self.cellList:
            if cell.type==3 or cell.type==4:
            #due to pressue from chemotaxis to vegf1, cell.volume is smaller that cell.target volume
            #in this simulation the offset is about 10 voxels.
                cell.targetVolume=64.0+10.0
                cell.lambdaVolume=20.0
            else:
                cell.targetVolume=32.0
                cell.lambdaVolume=20.0
    def step(self,mcs):
        fieldVEGF2=CompuCell.getConcentrationField(self.simulator,self.fieldNameVEGF2)
        fieldGlucose=CompuCell.getConcentrationField(self.simulator,self.fieldNameGlucose)
        print mcs
        
        for cell in self.cellList:
            #print cell.volume
            #NeoVascular
            if cell.type == 4:
                totalArea = 0
                pt=CompuCell.Point3D()
                pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                VEGFConcentration=fieldVEGF2.get(pt)
                
                cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
                for neighborSurfaceData in cellNeighborList:
                    #Check to ensure cell neighbor is not medium
                    if neighborSurfaceData.neighborAddress:
                        if neighborSurfaceData.neighborAddress.type == 3 or neighborSurfaceData.neighborAddress.type == 4:                            
                            #sum up common surface area of cell with its neighbors
                            totalArea+=neighborSurfaceData.commonSurfaceArea 
                            #print "  commonSurfaceArea:",neighborSurfaceData.commonSurfaceArea
                #print totalArea        
                if totalArea < 45:
                    #Growth rate equation
                    
                    cell.targetVolume+=2.0*VEGFConcentration/(0.01 + VEGFConcentration)
                    print "totalArea", totalArea,"cell growth rate: ", 2.0*VEGFConcentration/(0.01 + VEGFConcentration),"cell Volume: ", cell.volume
         
            #Proliferating Cells
            if cell.type == 1:
                
                pt=CompuCell.Point3D()
                pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                GlucoseConcentration=fieldGlucose.get(pt)
                # Proliferating Cells become Necrotic when GlucoseConcentration is low
                if  GlucoseConcentration < 0.001 and mcs>1000:
                    cell.type = 2
                    #set growth rate equation -- fastest cell cycle is 24hours or 1440 mcs--- 32voxels/1440mcs= 0.022 voxel/mcs
                    cell.targetVolume+=0.022*GlucoseConcentration/(0.05 + GlucoseConcentration)
                #print "growth rate: ", 0.044*GlucoseConcentration/(0.05 + GlucoseConcentration), "GlucoseConcentration", GlucoseConcentration

            #Necrotic Cells
            if cell.type == 2:
                #sNecrotic Cells shrink at a constant rate
                cell.targetVolume-=0.1
                
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableBase.__init__(self,_simulator, _frequency)
     
    def step(self,mcs):
        
        cells_to_divide=[]
          
        for cell in self.cellList:
            if cell.type == 1 and cell.volume>64:
                cells_to_divide.append(cell)
            if cell.type== 4 and cell.volume>128:
                cells_to_divide.append(cell)

                     
        for cell in cells_to_divide:

            self.divideCellRandomOrientation(cell)
            
    def updateAttributes(self):
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
        parentCell.targetVolume=parentCell.targetVolume/2
        parentCell.lambdaVolume=parentCell.lambdaVolume
        childCell.type=parentCell.type
        childCell.targetVolume=parentCell.targetVolume
        childCell.lambdaVolume=parentCell.lambdaVolume
          
                