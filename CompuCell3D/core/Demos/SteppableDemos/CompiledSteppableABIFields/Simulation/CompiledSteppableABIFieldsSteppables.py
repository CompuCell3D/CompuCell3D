from cc3d.core.PySteppables import SteppableBasePy
from cc3d.cpp import CompuCell


class FieldDrivenGrowthVerifierSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency=frequency)

    def start(self):
        print('Python field verifier start')

    def step(self, mcs):
        medium_counter = 0
        cell_counter = 0
        for x, y, z in self.every_pixel():
            cell = self.cell_field[x,y,z]
            if cell:
                cell_counter += 1
            else:
                medium_counter += 1
        print ("VERIFIER medium_counter=", medium_counter, "cell_counter=", cell_counter, "total=", medium_counter+cell_counter)


        fgf = self.field.FGF
        # for cell in self.cell_list_by_type(self.A):
        #     x = int(cell.xCOM)
        #     y = int(cell.yCOM)
        #     z = int(cell.zCOM)
        #     concentration = fgf.get(CompuCell.Point3D(x, y, z))
        #     print(
        #         f'Python field verifier step mcs={mcs} '
        #         f'cell_id={cell.id} fgf={concentration} targetVolume={cell.targetVolume}'
        #     )
        #     break
    def finish(self):
        print('Python field verifier finish')
