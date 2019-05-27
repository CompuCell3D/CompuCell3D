from PyPlugins import *

class VolumeEnergyFunctionPlugin(EnergyFunctionPy):

    def __init__(self,_energyWrapper):
        EnergyFunctionPy.__init__(self)
        self.energyWrapper=_energyWrapper
        self.vt=0.0
        self.lambda_v=0.0
    def setParams(self,_lambda,_targetVolume):
        self.lambda_v=_lambda;
        self.vt=_targetVolume
    def changeEnergy(self):
        energy=0
        newCell=self.energyWrapper.getNewCell()         
        oldCell=self.energyWrapper.getOldCell()         
        
        if(newCell):
            energy+=self.lambda_v*(1+2*(newCell.volume-self.vt))
        if(oldCell):
            energy+=self.lambda_v*(1-2*(oldCell.volume-self.vt))        
        return energy


