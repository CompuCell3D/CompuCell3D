from cc3d.core.PySteppables import SteppableBasePy
from cc3d.cpp import CompuCell


class CellFieldOperationsSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency=frequency)

    def start(self):
        print('Python field verifier start')

    def step(self, mcs):



        fgf = self.field.FGF
        for cell in self.cell_list:
            print("VERIFIER cell (id, type, targetVolume, lambdaVolume)=", cell.id, cell.type, cell.targetVolume, cell.lambdaVolume)


    def finish(self):
        print('Python field verifier finish')
