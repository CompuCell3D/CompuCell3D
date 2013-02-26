from PySteppables import *
import CompuCell
import CompuCellSetup
import sys



class InfoPrinterSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
        print "This function is called once before simulation"
        
    def step(self,mcs):
        print "This function is called every 10 MCS"
        for cell in self.cellList:
            print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
        if not ( mcs % 20 ):
            counter=0
            for cell in self.cellListByType(self.CONDENSING):
                print "BY TYPE CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
                counter+=1
            for cell in self.cellListByType(self.NONCONDENSING):
                print "BY TYPE CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
                counter+=1
                
            for cell in self.cellListByType(self.CONDENSING,self.NONCONDENSING):
                print "MULTI TYPE LIST - CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume                
                
            print "number of cells in typeInventory=",len(self.cellListByType)
            print "number of cells in the entire cell inventory=",len(self.cellList)                
            
        if mcs>500:
            CompuCellSetup.stopSimulation()


