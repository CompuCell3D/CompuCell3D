#Steppables 

"This module contains examples of certain more and less useful steppables written in Python"
from CompuCell import NeighborFinderParams


import CompuCell


    
from PySteppables import *

import sys
from random import random
import types
class ContactMultiCadSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.contactMultiCadPlugin=CompuCell.getContactMultiCadPlugin()
        self.inventory=self.simulator.getPotts().getCellInventory()
    def setTypeContactEnergyTable(self,_table):
        self.table=_table
        
    def start(self):
        invItr=CompuCell.STLPyIteratorCINV()
        invItr.initialize(self.inventory.getContainer())
        invItr.setToBegin()
        cell=invItr.getCurrentRef()
        cMultiCadDataAccessor=self.contactMultiCadPlugin.getContactMultiCadDataAccessorPtr()
        jVecItr=CompuCell.jVecPyItr()
        while (1):
            if invItr.isEnd():
                break
            cell=invItr.getCurrentRef()
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

            #itr=jVec.begin()
            #print "dereferenced itr",CompuCell.deref_stl_itr(itr)
            
            invItr.next()
            
        #sys.exit()
        #invItr.setToBegin()
        #while(not invItr.isEnd()):
            #cell=invItr.getCurrentRef()
            #jVec=cMultiCadDataAccessor.get(cell.extraAttribPtr).jVec
            ##jVec.set(0,self.table[cell.type])
	 #print "cell.id=",cell.id," vec[0]=",
            
	 #if isinstance(specificityObj,types.ListType):
                #jVec.set(0,(specificityObj[1]-specificityObj[0])*random())
                #jVec.set(1,(specificityObj[1]-specificityObj[0])*random())
            #else:
	     #jVec.set(0,specificityObj)
                #jVec.set(1,specificityObj)
            
            
        #sys.exit()
    def step(self,mcs):pass

