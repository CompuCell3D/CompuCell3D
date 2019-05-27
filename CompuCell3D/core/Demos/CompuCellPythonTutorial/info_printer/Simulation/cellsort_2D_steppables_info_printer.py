from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup


class InfoPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        print("This function is called once before simulation")

    def step(self, mcs):
        print("This function is called every 10 MCS")
        for cell in self.cell_list:
            print("CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
        if not (mcs % 20):
            counter = 0
            for cell in self.cell_list_by_type(self.CONDENSING):
                print("BY TYPE CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
                counter += 1
            for cell in self.cell_list_by_type(self.NONCONDENSING):
                print("BY TYPE CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
                counter += 1

            for cell in self.cell_list_by_type(self.CONDENSING, self.NONCONDENSING):
                print("MULTI TYPE LIST - CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)

            print("number of cells in typeInventory=", len(self.cell_list_by_type))
            print("number of cells in the entire cell inventory=", len(self.cell_list))

        if mcs > 500:
            CompuCellSetup.stopSimulation()
