from cc3d.core.PySteppables import *


class FocalPointPlasticityParams(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        list_of_cells = []
        for cell in self.cell_list:
            list_of_cells.append(cell)
        
        for i in range(len(list_of_cells)):
            for j in range(i+1, len(list_of_cells)):
                cell_i = list_of_cells[i]
                cell_j = list_of_cells[j]
                lambda_link = 10.0
                distance = 7.0
                max_distance = 20.0
                self.focalPointPlasticityPlugin.createFocalPointPlasticityLink(cell_i, cell_j, lambda_link, distance, max_distance)

