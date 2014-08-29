import CompuCellSetup
from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *
from random import random


class DiffusionFieldSteppable(SecretionBasePy): #IMPORTANT MAKE SURE YOU INHERIT FROM SecretionBasePy when you use steadyState solver and manage secretion from Python
    
    def __init__(self,_simulator,_frequency=1):
        SecretionBasePy.__init__(self,_simulator,_frequency)
        
    
    def start(self):  
        #initial condition for diffusion field    
        self.field=CompuCell.getConcentrationField(self.simulator,"FGF")        
        
        
#         import numpy as np
#         self.fieldNP = np.zeros(shape=(self.dim.x,self.dim.y,self.dim.z),dtype=np.float32)
#         fieldNP[:]=field        

        #a bit slow - will write faster version 
        secrConst=10
        for x,y,z in self.everyPixel(1,1,1):
            cell=self.cellField[x,y,z]
            if cell and cell.type==1:
                # notice for steady state solver we do not add secretion const to existing concentration
                # Also notice that secretion has to be negative (if we want positive secretion). This is how the solver is coded 
                self.field[x,y,z]=-secrConst    
            else:
                # for steady state solver all field pixels which do not secrete or uptake must me set to 0.0. This is how the solver works:    
                # non-zero value of the field at the pixel indicates secretion rate
                self.field[x,y,z]=0.0 
                
    def step(self, mcs):
        
        #a bit slow - will write faster version 
        secrConst=mcs
        for x,y,z in self.everyPixel(1,1,1):
            cell=self.cellField[x,y,z]
            if cell and cell.type==1:
                # notice for steady state solver we do not add secretion const to existing concentration
                # Also notice that secretion has to be negative (if we want positive secretion). This is how the solver is coded
                self.field[x,y,z]=-secrConst    
            else:
                # for steady state solver all field pixels which do not secrete or uptake must me set to 0.0. This is how the solver works:    
                # non-zero value of the field at the pixel indicates secretion rate
                self.field[x,y,z]=0.0 
        
        
