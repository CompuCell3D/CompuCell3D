from PySteppables import *
import CompuCell
import sys



class InfoPrinterSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def start(self):
        print "This function is called once before simulation"

    def step(self,mcs):
        print "This function is called every 10 MCS"
        for cell in self.cellList:
            print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
        if not ( mcs % 20 ):
            # self.cellListByType=CellListByType(1,self.inventory)
            self.cellListByType=CellListByType(self.inventory)
            self.cellListByType.initializeWithType(2)
            counter=0
            for cell in self.cellListByType:
                print "BY TYPE CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
                counter+=1
                
            print "number of cells in typeInventory=",len(self.cellListByType)
            print "number of cells in the entire cell inventory=",len(self.cellList)                




