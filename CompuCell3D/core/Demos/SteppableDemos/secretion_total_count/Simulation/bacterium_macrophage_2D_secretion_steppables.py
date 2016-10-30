from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss

NEW = True
            
class SecretionSteppable(SecretionBasePy):
    def __init__(self,_simulator,_frequency=1):
        SecretionBasePy.__init__(self,_simulator, _frequency)
        
    def step(self,mcs):
        attrSecretor=self.getFieldSecretor("ATTR")
        f = open('step_'+str(mcs).zfill(3)+'.dat','w')
        for cell in self.cellList:
            if cell.type==self.WALL:
                attrSecretor.secreteInsideCellAtBoundaryOnContactWith(cell,300,[self.WALL])
                attrSecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,300,[self.MEDIUM])
                res = attrSecretor.secreteInsideCellTotalCount(cell,300)
                print 'secreted  ', res.tot_amount, ' inside cell'
                attrSecretor.secreteInsideCellAtBoundaryTotalCount(cell,300)
                print 'secreted  ', res.tot_amount, ' inside cell at the boundary'
                attrSecretor.secreteOutsideCellAtBoundary(cell,500)
                attrSecretor.secreteInsideCellAtCOM(cell,300)        
                
                res = attrSecretor.uptakeInsideCellTotalCount(cell,3,0.1)
                print 'Total uptake inside cell ', res.tot_amount

                attrSecretor.uptakeInsideCellAtBoundaryOnContactWith(cell,3,0.1,[self.MEDIUM])
                attrSecretor.uptakeOutsideCellAtBoundaryOnContactWith(cell,3,0.1,[self.MEDIUM])

                res = attrSecretor.uptakeInsideCellAtBoundaryTotalCount(cell,3,0.1)
                print 'Total uptake inside cell at the boundary ', res.tot_amount
                attrSecretor.uptakeOutsideCellAtBoundary(cell,3,0.1)
                attrSecretor.uptakeInsideCellAtCOM(cell,3,0.1)        
                


        field=self.getConcentrationField('ATTR')
        
        for x,y,z in self.everyPixel():
            
            print >>f ,(x,y,z,field[x,y,z])
        
        f.close()
        
        
                
        

