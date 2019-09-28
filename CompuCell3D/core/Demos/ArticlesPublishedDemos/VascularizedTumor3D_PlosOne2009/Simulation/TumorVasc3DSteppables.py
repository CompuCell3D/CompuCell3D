from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.area_thresh = None
        self.nutrient_thresh = None
        self.necrotic_thresh = None

    def set_params(self, area_thresh=0, nutrient_thresh=0, necrotic_thresh=0):

        self.area_thresh = area_thresh
        self.nutrient_thresh = nutrient_thresh
        self.necrotic_thresh = necrotic_thresh

    def start(self):

        for cell in self.cellList:
            if cell.type == 4 or cell.type == 5 or cell.type == 6:
                cell.targetVolume = 60
                cell.lambdaVolume = 13.0
                cell.targetSurface = 150
                cell.lambdaSurface = 3.0
            else:
                cell.targetVolume = 33.0
                cell.lambdaVolume = 10.0
                cell.targetSurface = 90.0
                cell.lambdaSurface = 2

    def step(self, mcs):

        field_neo_vasc = self.field.VEGF2
        field_malig = self.field.Oxygen

        for cell in self.cellList:

            # Inactive neovascular differentiation
            if cell.type == 6:
                total_area = 0
                x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                z = int(round(cell.zCM / max(float(cell.volume), 0.001)))

                concentration = field_neo_vasc[x, y, z]
                if concentration > 0.5:

                    neighbor_list = self.get_cell_neighbor_data_list(cell)
                    for neighbor, common_surface_area in neighbor_list:
                        if neighbor:
                            if neighbor.type in [5, 6, 7]:
                                total_area += common_surface_area
                    print(cell.type, total_area)
                    if total_area < 70:
                        # Growth rate equation
                        cell.targetVolume += 0.06 * concentration / (0.5 + concentration)
                        cell.targetSurface += 0.15 * concentration / (0.5 + concentration)

            ## Active neovascular growth
            if cell.type == 4:
                total_area = 0

                x = int(round(cell.xCM / max(float(cell.volume), 0.00000001)))
                y = int(round(cell.yCM / max(float(cell.volume), 0.00000001)))
                z = int(round(cell.zCM / max(float(cell.volume), 0.00000001)))

                concentration = field_neo_vasc[x, y, z]

                if concentration > 0.5:
                    neighbor_list = self.get_cell_neighbor_data_list(cell)
                    for neighbor, common_surface_area in neighbor_list:
                        if neighbor:
                            if neighbor.type in [5, 6, 7]:
                                total_area += common_surface_area

                    if total_area < 50:
                        # Growth rate equation

                        cell.targetVolume += 0.06 * concentration / (0.5 + concentration)
                        cell.targetSurface += 0.15 * concentration / (0.5 + concentration)

            # Malignat and Hypoxic Cells growth
            if cell.type == 1 or cell.type == 2:

                x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                z = int(round(cell.zCM / max(float(cell.volume), 0.001)))

                concentration2 = field_malig[x, y, z]

                # switch to Hypoxic cell type
                if concentration2 < self.nutrient_thresh and mcs > 100:
                    cell.type = 2

                # switch to Necrotic cell type
                if concentration2 < self.necrotic_thresh and mcs > 100:
                    cell.type = 3

                # set growth rate equation
                if mcs > 100:
                    cell.targetVolume += 0.04 * concentration2 / (10 + concentration2)
                    cell.targetSurface += 0.12 * concentration2 / (10 + concentration2)

            # Hypoxic Cells
            if cell.type == 2:
                x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                z = int(round(cell.zCM / max(float(cell.volume), 0.001)))

                concentration3 = field_malig[x, y, z]

                # switch to Necrotic cell type
                if concentration3 < self.necrotic_thresh and mcs > 100:
                    cell.type = 3
                # switch to Normal cell type
                if concentration3 > self.nutrient_thresh:
                    cell.type = 1

            # Necrotic Cells
            if cell.type == 3:
                # set growth rate equation
                cell.targetVolume -= 0.5
                cell.lambdaSurface = 0


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.doublingVolumeDict = None

    def set_params(self, doublingVolumeDict):
        self.doublingVolumeDict = doublingVolumeDict

    def step(self, mcs):

        cells_to_divide = []
        for cell in self.cell_list_by_type(self.NORMAL,
                                           self.HYPOXIC,
                                           self.ACTIVENEOVASCULAR,
                                           self.INACTIVENEOVASCULAR):
            try:
                doubling_volume = self.doublingVolumeDict[cell.type]
            except KeyError:
                continue

            if cell.volume > doubling_volume:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)

    def update_attributes(self):

        if self.parentCell.type == 1 or self.parentCell.type == 2:
            self.childCell.type = 1
            self.childCell.targetVolume = 33
            self.childCell.lambdaVolume = 10
            self.childCell.targetSurface = 90
            self.childCell.lambdaSurface = 2
            self.parentCell.targetVolume = 33
            self.parentCell.lambdaVolume = 10
            self.parentCell.targetSurface = 90
            self.parentCell.lambdaSurface = 2

        ## Mitosis of ActiveNeovascular and InactiveNeovascular cells
        if self.parentCell.type == 6 or self.parentCell.type == 4:
            self.childCell.type = 4
            self.childCell.targetVolume = 60
            self.childCell.lambdaVolume = 13
            self.childCell.targetSurface = 150
            self.childCell.lambdaSurface = 3
            self.parentCell.targetVolume = 60
            self.parentCell.lambdaVolume = 13
            self.parentCell.targetSurface = 150
            self.parentCell.lambdaSurface = 3
