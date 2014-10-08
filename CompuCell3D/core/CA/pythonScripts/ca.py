import math

print 'This is CA Python run module'

import CA

caManager=CA.CAManager()
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

