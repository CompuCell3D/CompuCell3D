from PySteppables import *
import CompuCell
import sys



class FluctuationAmplitude(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.quarters=[[0,0,50,50],[0,50,50,100],[50,50,100,100],[50,0,100,50]]
        self.steppableCallCounter=0
        
    def step(self, mcs):        
        
        
        quarterIndex=self.steppableCallCounter % 4
        quarter=self.quarters[quarterIndex]
        for cell in self.cellList:
            
            if cell.xCOM>=quarter[0] and cell.yCOM>=quarter[1] and cell.xCOM<quarter[2] and cell.yCOM<quarter[3]:
                cell.fluctAmpl=50                
            else:
                #this means CompuCell3D will use globally defined FluctuationAmplitude
                cell.fluctAmpl=-1                 
        self.steppableCallCounter+=1

 