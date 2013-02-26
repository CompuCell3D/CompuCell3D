#Steppables 

import CompuCell
from PySteppables import *

import sys
from random import random
import types


class ContactMultiCadSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.contactMultiCadPlugin=CompuCell.getContactMultiCadPlugin()

    def setTypeContactEnergyTable(self,_table):
        self.table=_table
        
    def start(self):
        
        cMultiCadDataAccessor=self.contactMultiCadPlugin.getContactMultiCadDataAccessorPtr()

        for cell in self.cellList:
            
            cMultiCadData=cMultiCadDataAccessor.get(cell.extraAttribPtr)
            #jVec.set(0,self.table[cell.type])
            specificityObj=self.table[cell.type]
            print "cell.type=",cell.type," specificityObj=",specificityObj
            if isinstance(specificityObj,types.ListType):
                cMultiCadData.assignValue(0,(specificityObj[1]-specificityObj[0])*random())
                cMultiCadData.assignValue(1,(specificityObj[1]-specificityObj[0])*random())
            else:
                cMultiCadData.assignValue(0,specificityObj)
                cMultiCadData.assignValue(1,specificityObj/2)

