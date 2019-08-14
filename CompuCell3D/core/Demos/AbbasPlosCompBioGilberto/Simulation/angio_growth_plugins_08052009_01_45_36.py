####
#### The simulation code is compatible with CompuCell3D ver 3.3.1
####

from CompuCell import MitosisSimplePlugin      
from PyPlugins import *
from PySteppables import CellList
from CompuCell import NeighborFinderParams
import time,sys

class MitosisPyPluginBase(StepperPy,Field3DChangeWatcherPy):
   def __init__(self,_simulator,_changeWatcherRegistry,_stepperRegistry):

   
      Field3DChangeWatcherPy.__init__(self,_changeWatcherRegistry)
      self.simulator=_simulator
      self.mitosisPlugin=MitosisSimplePlugin()
      self.mitosisPlugin.setPotts(self.simulator.getPotts())
      self.mitosisPlugin.turnOn()
      self.mitosisPlugin.init(self.changeWatcher.sim)
      self.counter=0
      self.mitosisFlag=0
      self.doublingVolumeDict=0
      _changeWatcherRegistry.registerPyChangeWatcher(self)
      _stepperRegistry.registerPyStepper(self)        
   

      
      
   def setPotts(self,potts):
      self.mitosisPlugin.setPotts(potts)
   
   def setDoublingVolume(self,_doublingVolume):
      self.doublingVolume=_doublingVolume;
      self.mitosisPlugin.setDoublingVolume(self.doublingVolume)
   
   def setCellDoublingVolume(self,_doublingVolumeDict):
      self.doublingVolumeDict=_doublingVolumeDict;
      for i in self.doublingVolumeDict.keys():
         print self.doublingVolumeDict[i]

   def field3DChange(self):
      cell = self.changeWatcher.newCell
      if cell and self.doublingVolumeDict.has_key(cell.type) and cell.volume>self.doublingVolumeDict[cell.type]:
         print "Type: ", cell.type, " Doubling Volume: ", self.doublingVolumeDict[cell.type], " Current Volume: ", cell.volume
         self.setDoublingVolume(self.doublingVolumeDict[cell.type])
         self.mitosisPlugin.field3DChange(self.changeWatcher.changePoint,self.changeWatcher.newCell,self.changeWatcher.newCell)
         self.mitosisFlag=1
         
   def step(self):
      if self.mitosisFlag:
         print "ABOUT TO DO MITOSIS"
         self.mitosisFlag=self.mitosisPlugin.doMitosis()
         self.childCell=self.mitosisPlugin.getChildCell()
         self.parentCell=self.mitosisPlugin.getParentCell()
         self.updateAttributes()
         self.mitosisFlag=0
         
   def updateAttributes(self):
      self.childCell.targetVolume=self.parentCell.targetVolume
      self.childCell.lambdaVolume=self.parentCell.lambdaVolume
      self.childCell.type=self.parentCell.type


class MitosisPyPlugin(MitosisPyPluginBase):
   def __init__(self , _simulator , _changeWatcherRegistry , _stepperRegistry):
      MitosisPyPluginBase.__init__(self,_simulator,_changeWatcherRegistry,_stepperRegistry)


   def updateAttributes(self):
## Mitosis of normal tumor and hypoxic cells    
      if self.parentCell.type==1 or self.parentCell.type==2:
         self.childCell.type=1
	 self.childCell.targetVolume=33
         self.childCell.lambdaVolume=10
         self.childCell.targetSurface=90
         self.childCell.lambdaSurface=2
         self.parentCell.targetVolume=33
         self.parentCell.lambdaVolume=10
         self.parentCell.targetSurface=90
         self.parentCell.lambdaSurface=2
## Mitosis of ActiveNeovascular and InactiveNeovascular cells
      if self.parentCell.type==6 or self.parentCell.type==4:
         self.childCell.type=4
	 self.childCell.targetVolume=60
         self.childCell.lambdaVolume=13
         self.childCell.targetSurface=150
         self.childCell.lambdaSurface=3
         self.parentCell.targetVolume=60
         self.parentCell.lambdaVolume=13
         self.parentCell.targetSurface=150
         self.parentCell.lambdaSurface=3
