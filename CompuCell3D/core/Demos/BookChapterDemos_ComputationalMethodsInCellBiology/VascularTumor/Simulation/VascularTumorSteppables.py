from cc3d.core.PySteppables import *


def ir(x):
    """
    Rounds floating point to thew nearest integer ans returns integer
    :param x: {float} num to round
    :return:
    """
    return int(round(x))


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1, ):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cellList:
            if cell.type == self.VASCULAR or cell.type == self.NEOVASCULAR:
                # due to pressue from chemotaxis to vegf1, cell.volume is smaller that cell.target volume
                # in this simulation the offset is about 10 voxels.
                cell.targetVolume = 64.0 + 10.0
                cell.lambdaVolume = 20.0
            else:
                cell.targetVolume = 32.0
                cell.lambdaVolume = 20.0

    def step(self, mcs):
        field_vegf2 = self.field.VEGF2
        field_glucose = self.field.Glucose

        for cell in self.cell_list:
            # NeoVascular
            if cell.type == self.NEOVASCULAR:
                total_area = 0

                vegf_concentration = field_vegf2[ir(cell.xCOM), ir(cell.yCOM), ir(cell.zCOM)]

                neighbor_list = self.get_cell_neighbor_data_list(cell)
                for neighbor, common_surface_area in neighbor_list:
                    # Check to ensure cell neighbor is not medium
                    if neighbor:
                        if neighbor.type == self.VASCULAR or neighbor.type == self.NEOVASCULAR:
                            # sum up common surface area of cell with its neighbors
                            total_area += common_surface_area

                if total_area < 45:
                    # Growth rate equation
                    cell.targetVolume += 2.0 * vegf_concentration / (0.01 + vegf_concentration)
                    print("total_area", total_area, "cell growth rate: ",
                          2.0 * vegf_concentration / (0.01 + vegf_concentration), "cell Volume: ", cell.volume)

            # Proliferating Cells
            if cell.type == self.PROLIFERATING:

                glucose_concentration = field_glucose[ir(cell.xCOM), ir(cell.yCOM), ir(cell.zCOM)]

                # Proliferating Cells become Necrotic when glucose_concentration is low
                if glucose_concentration < 0.001 and mcs > 1000:
                    cell.type = self.NECROTIC
                    # set growth rate equation -- fastest cell cycle is 24hours or 1440 mcs
                    # --- 32voxels/1440mcs= 0.022 voxel/mcs
                cell.targetVolume += 0.022 * glucose_concentration / (0.05 + glucose_concentration)

            # Necrotic Cells
            if cell.type == self.NECROTIC:
                # Necrotic Cells shrink at a constant rate
                cell.targetVolume -= 0.1


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def step(self, mcs):

        cells_to_divide = []

        for cell in self.cell_list:
            if cell.type == self.PROLIFERATING and cell.volume > 64:
                cells_to_divide.append(cell)
            if cell.type == self.NEOVASCULAR and cell.volume > 128:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0
        self.clone_parent_2_child()
