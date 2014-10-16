import sys,os,time
from random import random,randint,shuffle


print ''
print sys.path
print os.environ['PYTHONPATH']


import math

print 'This is CA Python run module'

import CA
# import Simulation.SimulationThreadCA
from Simulation.SimulationThreadCA import SimulationThreadCA

from CAPySteppables import *


class MitosisSteppable(CAPySteppableBase):
    def __init__(self,_caManager,_frequency=1):
        CAPySteppableBase.__init__(self,_caManager,_frequency)
    def start(self):
        if self.dim.z>1:
            cell=caManager.createAndPositionCellS(CA.Point3D(self.dim.x/2,self.dim.y/2,self.dim.z/2))
        else:
            cell=caManager.createAndPositionCellS(CA.Point3D(self.dim.x/2,self.dim.y/2,0))
        cell.type=randint(1,2)    
        
        # # # pt=CA.Point3D()
        # # # for x in xrange(self.dim.x/2):
            # # # for y in xrange(self.dim.y/2):
                # # # for z in xrange(self.dim.z/2):
                    # # # pt.x=x
                    # # # pt.y=y
                    # # # pt.z=z
                    # # # cell=caManager.createAndPositionCellS(pt)
                    # # # cell.type=1
                    # # # print 'pt=',pt,' cell=',cell
                    
    def step(self,mcs):

        # a=CellList(None)    
        bs = self.caManager.getBoundaryStrategy()
        if self.dim.z>1:
            self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(3)
        else:
            self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(2)
            
        print '\n\n\n\n maxNeighborIdx=',self.maxNeighborIdx
        cellVector = self.inventory.generateCellVector()
        print 'dir(cellVector)=',dir(cellVector)
        print 'cellVector.size()=',cellVector.size()
        idx = randint(0,cellVector.size()-1)
        cell = cellVector[idx]
        print 'idx=',idx,'cell.id=',cell.id
        # x_rand=randint(0,self.dim.x-1)
        # y_rand=randint(0,self.dim.y-1)
        # cell=caManager.createAndPositionCellS(CA.Point3D(x_rand,y_rand,0))
        # cell.type=randint(1,2)    
        
        # # # pt = CA.Point3D(randint(0,self.dim.x/2),randint(0,self.dim.y/2),randint(0,self.dim.z/2))

        # # # cellStackLocal = self.cellField.get(pt)
        
        # # # if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
            
            # # # cell = caManager.createAndPositionCellS(pt)
            # # # cell.type = randint(1,2)
        
        
        # # # return 
        
        cells_to_divide=[]    
        for cell in self.cellList:
            # print 'cell.id',cell.id,' cell.type=',cell.type,'cell.xCOM=',cell.xCOM,' yCOM=',cell.yCOM
            cells_to_divide.append(cell)
        
        shuffle(cells_to_divide)
        for cell in cells_to_divide:
            # print 'cell.id=',cell.id
            print 'xCOM=',cell.xCOM,' yCOM=',cell.yCOM, 'zCOM=',cell.zCOM
            cellStack = self.cellField.get(CA.Point3D(cell.xCOM,cell.yCOM,cell.zCOM))
            comPt = CA.Point3D(cell.xCOM,cell.yCOM,cell.zCOM)
            
            randOffsetIdx = randint(0,self.maxNeighborIdx)
            
            neighbor=bs.getNeighborDirect( comPt , randOffsetIdx)
            print 'randOffsetIdx=',randOffsetIdx,' neighbor.pt=',neighbor.pt
            if not neighbor.distance:
                continue
                
            cellStackLocal=self.cellField.get(neighbor.pt)
         
            
            if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                
                cell = caManager.createAndPositionCellS(neighbor.pt)
                cell.type = randint(1,2)                
            
            # if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                
                # cell = caManager.createAndPositionCellS(neighbor.pt)
                # cell.type = randint(1,2)
                
            # # # # if not cellStack.isFull():
                # # # # cell = caManager.createAndPositionCellS(comPt)
                # # # # cell.type = randint(1,2)
            # # # # else:
                # # # # randOffsetIdx = randint(0,self.maxNeighborIdx)
                # # # # neighbor=bs.getNeighborDirect( comPt , randOffsetIdx)
                # # # # if not neighbor.distance:
                    # # # # continue
                    
                # cellStackLocal=self.cellField.get(neighbor.pt)
                
                # if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                    
                    # cell = caManager.createAndPositionCellS(neighbor.pt)
                    # cell.type = randint(1,2)                    
            
    # def step(self,mcs):

        # # a=CellList(None)    
        # bs = self.caManager.getBoundaryStrategy()
        # if self.dim.z>1:
            # self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(3)
        # else:
            # self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(2)
            
        # print '\n\n\n\n maxNeighborIdx=',self.maxNeighborIdx
        # cellVector = self.inventory.generateCellVector()
        # print 'dir(cellVector)=',dir(cellVector)
        # print 'cellVector.size()=',cellVector.size()
        # idx = randint(0,cellVector.size()-1)
        # cell = cellVector[idx]
        # print 'idx=',idx,'cell.id=',cell.id
        # # x_rand=randint(0,self.dim.x-1)
        # # y_rand=randint(0,self.dim.y-1)
        # # cell=caManager.createAndPositionCellS(CA.Point3D(x_rand,y_rand,0))
        # # cell.type=randint(1,2)    

        # cells_to_divide=[]    
        # for cell in self.cellList:
            # # print 'cell.id',cell.id,' cell.type=',cell.type,'cell.xCOM=',cell.xCOM,' yCOM=',cell.yCOM
            # cells_to_divide.append(cell)
        
        # for cell in cells_to_divide:
            
            # cellStack = self.cellField.get(CA.Point3D(cell.xCOM,cell.yCOM,cell.zCOM))
            # comPt = CA.Point3D(cell.xCOM,cell.yCOM,cell.zCOM)
            
            # randOffsetIdx = randint(0,self.maxNeighborIdx)
            
            # neighbor=bs.getNeighborDirect( comPt , randOffsetIdx)
            # print 'randOffsetIdx=',randOffsetIdx,' neighbor.pt=',neighbor.pt
            # if not neighbor.distance:
                # continue
                
            # cellStackLocal=self.cellField.get(neighbor.pt)
            
            # if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                
                # cell = caManager.createAndPositionCellS(neighbor.pt)
                # cell.type = randint(1,2)
                
            # # # # if not cellStack.isFull():
                # # # # cell = caManager.createAndPositionCellS(comPt)
                # # # # cell.type = randint(1,2)
            # # # # else:
                # # # # randOffsetIdx = randint(0,self.maxNeighborIdx)
                # # # # neighbor=bs.getNeighborDirect( comPt , randOffsetIdx)
                # # # # if not neighbor.distance:
                    # # # # continue
                    
                # cellStackLocal=self.cellField.get(neighbor.pt)
                
                # if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                    
                    # cell = caManager.createAndPositionCellS(neighbor.pt)
                    # cell.type = randint(1,2)
                
    


caManager=CA.CAManager()
caManager.setNumSteps(100)
print 'caManager=',dir(caManager)

from CAPyUtils import DemoSteppable
        
ds=DemoSteppable(caManager)        





caManager.createCellField(CA.Dim3D(100,100,100))
caManager.setCellCarryingCapacity(2)
cellField=caManager.getCellField()
fieldDim = cellField.getDim()
print 'dim of cell field=',cellField.getDim()

cellFieldS=caManager.getCellFieldS()
print 'cellFieldS=',cellFieldS

print 'cellFieldS.getDim()=',cellFieldS.getDim()


# cell=caManager.createAndPositionCellS(CA.Point3D(1,1,0))
# cell.type=1

# print 'AFTER STEP caManager.getCellInventory().getSize()=',caManager.getCellInventory().getSize()

# cell1=caManager.createAndPositionCellS(CA.Point3D(1,1,0))
# cell1.type=1


# cell2=caManager.createAndPositionCellS(CA.Point3D(1,1,0))
# cell2.type=1

# print 'AFTER STEP caManager.getCellInventory().getSize()=',caManager.getCellInventory().getSize()
# # time.sleep(2)

print dir(CA)    
#REGISTERING COM MONITOR
comMonitor = CA.CenterOfMassMonitor()    
comMonitor.init(caManager)
print 'comMonitor=',comMonitor


caManager.registerCellFieldChangeWatcher(comMonitor)


# try:
    # cell=caManager.createAndPositionCell(CA.Point3D(1,1,0))
    # print 'cell.type=',cell.type,' id=',cell.id
    # cell.type=1
    # print 'AFTER ASIGNNMENT cell.type=',cell.type,' id=',cell.id
    
    
# except RuntimeError,e:
    # print e
    

# # # caManager.positionCell(CA.Point3D(1,1,1),cell)
    
# # # # cellField.set(CA.Point3D(1,1,1),cell)

# # # # cell1=cellField.get(CA.Point3D(9,1,1))
# # # # print 'cell1.id =',cell1.id


# # # cell2=caManager.createAndPositionCell(CA.Point3D(5,1,3))
# # # print '\n\ncell2.id=',cell2.id
# # # ds.step(1)

# # # caManager.positionCell(CA.Point3D(5,1,3),cell)
# # # print '\nafter position 913 '


# # # cell1=cellField.get(CA.Point3D(1,1,1))
# # # if cell1:
    # # # print 'cell1.id =',cell1.id
# # # else:
    # # # print 'CELL OBJ IS NONE' 


    
# # # cell1=cellField.get(CA.Point3D(5,1,3))
# # # if cell1:
    # # # print 'cell1.id =',cell1.id
# # # else:
    # # # print 'CELL OBJ IS NONE' 
    

# # # cell1=cellField.get(CA.Point3D(1,1,1))
# # # if cell1:
    # # # print 'cell1.id =',cell1.id
# # # else:
    # # # print 'CELL OBJ IS NONE' 

# # # print 'cell inventory size=',caManager.getCellInventory().getSize()



steppableRegistry=CompuCellSetup.getSteppableRegistry()
mitosisSteppable=MitosisSteppable(caManager)
steppableRegistry.registerSteppable(mitosisSteppable)



ds.step(2)


simthread=CompuCellSetup.simulationThreadObject

simthread.stopped=False
simthread.setSimulator(caManager)


if not steppableRegistry is None:    
    steppableRegistry.start()

simthread.postStartInit()
simthread.waitForPlayerTaskToFinish()


# # # if not steppableRegistry is None:
    # # # steppableRegistry.start()

beginingStep=0
i=beginingStep

print '\n\n\n\nBEFORE MAIN LOOP CA.PY'
print 'caManager.getCellInventory().getSize()=',caManager.getCellInventory().getSize()

while True:
    simthread.beforeStep(i)
    
    # cell=caManager.createAndPositionCell(CA.Point3D(1,1,0))

    
    caManager.step(i)
    
    if not steppableRegistry is None:
        steppableRegistry.step(i)
    # mitosisSteppable.step(i)
    
    # cell=caManager.createAndPositionCellS(CA.Point3D(randint(0,fieldDim.x-1),randint(0,fieldDim.y-1),0))
    # # # cell=caManager.createAndPositionCellS(CA.Point3D(1,2,0))
    # # # cell.type=randint(1,10)
    # # # print '\n\nstep = ', i
    # # # print 'simthread=',simthread
    # # # print 'AFTER STEP caManager.getCellInventory().getSize()=',caManager.getCellInventory().getSize()
    if simthread.getStopSimulation() or CompuCellSetup.userStopSimulationFlag:
        runFinishFlag=False;
        break 
    simthread.loopWork(i)
    simthread.loopWorkPostEvent(i)
    screenUpdateFrequency = simthread.getScreenUpdateFrequency()
   
    i+=1        
    if i>=caManager.getNumSteps():
        break    
    
caManager.cleanAfterSimulation()
# sim.unloadModules()
print "CALLING UNLOAD MODULES NEW PLAYER"
if simthread is not None:
    simthread.sendStopSimulationRequest()
    print '\n\n\n\n\n\n\n SENDING HERE SIMULATION FINISHED REQUEST'
    simthread.simulationFinishedPostEvent(True)

    
# import time
# time.sleep(3)

import CompuCellSetup
print 'CompuCellSetup.playerType=',CompuCellSetup.playerType
print 'CompuCellSetup.playerModel=',CompuCellSetup.playerModel
print 'CompuCellSetup.simulationThreadObject=',CompuCellSetup.simulationThreadObject


