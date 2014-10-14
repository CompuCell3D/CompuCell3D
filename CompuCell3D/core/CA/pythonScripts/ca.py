import sys,os
print ''
print sys.path
print os.environ['PYTHONPATH']


import math

print 'This is CA Python run module'

import CA
# import Simulation.SimulationThreadCA
from Simulation.SimulationThreadCA import SimulationThreadCA

caManager=CA.CAManager()
caManager.setNumSteps(10)
print 'caManager=',dir(caManager)

from CAPyUtils import DemoSteppable
        
ds=DemoSteppable(caManager)        




caManager.createCellField(CA.Dim3D(10,10,10))
cellField=caManager.getCellField()
print 'dim of cell field=',cellField.getDim()


print dir(CA)    
#REGISTERING COM MONITOR
comMonitor = CA.CenterOfMassMonitor()    
comMonitor.init(caManager)
print 'comMonitor=',comMonitor


caManager.registerCellFieldChangeWatcher(comMonitor)


try:
    cell=caManager.createAndPositionCell(CA.Point3D(9,1,1))
    print 'cell.type=',cell.type,' id=',cell.id
    cell.type=1
    print 'AFTER ASIGNNMENT cell.type=',cell.type,' id=',cell.id
    
    
except RuntimeError,e:
    print e
    

caManager.positionCell(CA.Point3D(1,1,1),cell)
    
# cellField.set(CA.Point3D(1,1,1),cell)

# cell1=cellField.get(CA.Point3D(9,1,1))
# print 'cell1.id =',cell1.id


cell2=caManager.createAndPositionCell(CA.Point3D(9,1,3))
print '\n\ncell2.id=',cell2.id
ds.step(1)

caManager.positionCell(CA.Point3D(9,1,3),cell)
print '\nafter position 913 '


cell1=cellField.get(CA.Point3D(1,1,1))
if cell1:
    print 'cell1.id =',cell1.id
else:
    print 'CELL OBJ IS NONE' 


    
cell1=cellField.get(CA.Point3D(9,1,3))
if cell1:
    print 'cell1.id =',cell1.id
else:
    print 'CELL OBJ IS NONE' 
    

cell1=cellField.get(CA.Point3D(1,1,1))
if cell1:
    print 'cell1.id =',cell1.id
else:
    print 'CELL OBJ IS NONE' 

print 'cell inventory size=',caManager.getCellInventory().getSize()



ds.step(2)


simthread=CompuCellSetup.simulationThreadObject

simthread.setSimulator(caManager)

simthread.postStartInit()
simthread.waitForPlayerTaskToFinish()

beginingStep=0
i=beginingStep

while True:
    simthread.beforeStep(i)
    caManager.step(i)
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
    simthread.simulationFinishedPostEvent(True)

    
# import time
# time.sleep(3)

import CompuCellSetup
print 'CompuCellSetup.playerType=',CompuCellSetup.playerType
print 'CompuCellSetup.playerModel=',CompuCellSetup.playerModel
print 'CompuCellSetup.simulationThreadObject=',CompuCellSetup.simulationThreadObject


