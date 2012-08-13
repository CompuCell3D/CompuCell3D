#Steppables

import CompuCell
import PlayerPython
from PySteppables import *


class PixelTrackerSteppable(SteppableBasePy):
   def __init__(self,_simulator,_frequency=10):
      SteppableBasePy.__init__(self,_simulator,_frequency)

   def start(self):pass

   def step(self,mcs):
      for cell in self.cellList:
         if cell.type==2:
            pixelList=self.getCellPixelList(cell)
            for pixelTrackerData in pixelList:
               print "pixel of cell id=",cell.id," type:",cell.type, " = ",pixelTrackerData.pixel," number of pixels=",pixelList.numberOfPixels()


