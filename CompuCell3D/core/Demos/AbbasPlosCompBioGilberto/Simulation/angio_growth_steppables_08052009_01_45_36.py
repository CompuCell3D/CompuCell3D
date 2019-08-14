from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.areaThresh = None
        self.nutrientThresh = None
        self.necroticThresh = None
        self.fieldNameNeoVascular = 'VEGF2'
        self.fieldNameNormal = 'Oxygen'
        # self.output_file = open("CellDiffusionData_08052009_01_45_36.txt",'w')

    def set_params(self, areaThresh=0, nutrientThresh=0, necroticThresh=0):

        self.areaThresh = areaThresh
        self.nutrientThresh = nutrientThresh
        self.necroticThresh = necroticThresh

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

        fieldNeoVasc = CompuCell.getConcentrationField(self.simulator, self.fieldNameNeoVascular)
        fieldMalig = CompuCell.getConcentrationField(self.simulator, self.fieldNameNormal)

        for cell in self.cellList:

            # Inactive neovascular differentiation
            if cell.type == 6:
                totalArea = 0
                pt = CompuCell.Point3D()
                pt.x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                pt.y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                pt.z = int(round(cell.zCM / max(float(cell.volume), 0.001)))
                concentration = fieldNeoVasc.get(pt)
                if concentration > 0.5:

                    # cellNeighborList = CellNeighborListAuto(self.nTrackerPlugin, cell)
                    neighbor_list = self.get_cell_neighbor_data_list(cell)
                    for neighbor, common_surface_area in neighbor_list:
                        if neighbor:
                            if neighbor.type in [5, 6, 7]:
                                totalArea += common_surface_area

                    # for neighborSurfaceData in cellNeighborList:
                    #     # Check to ensure cell neighbor is not medium
                    #     if neighborSurfaceData.neighborAddress:
                    #         if (neighborSurfaceData.neighborAddress.type == 5
                    #                 or neighborSurfaceData.neighborAddress.type == 6
                    #                 or neighborSurfaceData.neighborAddress.type == 7):
                    #             # sum up common surface area of cell with its neighbors
                    #             totalArea += neighborSurfaceData.commonSurfaceArea

                    print(cell.type, totalArea)
                    if totalArea < 70:
                        # Growth rate equation
                        cell.targetVolume += 0.06 * concentration / (0.5 + concentration)
                        cell.targetSurface += 0.15 * concentration / (0.5 + concentration)
                        # print 0.02*concentration/(0.5 + concentration)+0.04

            ## Active neovascular growth
            if cell.type == 4:
                totalArea = 0
                pt = CompuCell.Point3D()
                pt.x = int(round(cell.xCM / max(float(cell.volume), 0.00000001)))
                pt.y = int(round(cell.yCM / max(float(cell.volume), 0.00000001)))
                pt.z = int(round(cell.zCM / max(float(cell.volume), 0.00000001)))
                concentration = fieldNeoVasc.get(pt)
                if concentration > 0.5:
                    neighbor_list = self.get_cell_neighbor_data_list(cell)
                    for neighbor, common_surface_area in neighbor_list:
                        if neighbor:
                            if neighbor.type in [5, 6, 7]:
                                totalArea += common_surface_area

                    # cellNeighborList = CellNeighborListAuto(self.nTrackerPlugin, cell)
                    # for neighborSurfaceData in cellNeighborList:
                    #     # Check to ensure cell neighbor is not medium
                    #     if neighborSurfaceData.neighborAddress:
                    #         if (neighborSurfaceData.neighborAddress.type == 5
                    #                 or neighborSurfaceData.neighborAddress.type == 7
                    #                 or neighborSurfaceData.neighborAddress.type == 6):
                    #             # sum up common surface area of cell with its neighbors
                    #             totalArea += neighborSurfaceData.commonSurfaceArea
                    #             # print "concentration: ", concentration,"  commonSurfaceArea:",neighborSurfaceData.commonSurfaceArea
                    # print cell.type,totalArea
                    if totalArea < 50:
                        # Growth rate equation
                        # print cell.type,"##surface area",cell.surface,"##cell volume:",cell.volume,"##cell target volume:",cell.targetVolume,"##common surface area:",totalArea
                        cell.targetVolume += 0.06 * concentration / (0.5 + concentration)
                        cell.targetSurface += 0.15 * concentration / (0.5 + concentration)
                        ##print 0.02*concentration/(0.5 + concentration)+0.04

            # Malignat and Hypoxic Cells growth
            if cell.type == 1 or cell.type == 2:
                # print cell.volume

                pt = CompuCell.Point3D()
                pt.x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                pt.y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                pt.z = int(round(cell.zCM / max(float(cell.volume), 0.001)))
                # self.output_file.write("%f %f %f " %(cell.xCM/cell.volume, cell.yCM/cell.volume,cell.zCM/cell.volume))

                concentration2 = fieldMalig.get(pt)
                # switch to Hypoxic cell type
                if (concentration2 < self.nutrientThresh and mcs > 100):
                    cell.type = 2

                # switch to Necrotic cell type
                if concentration2 < self.necroticThresh and mcs > 100:
                    cell.type = 3

                # set growth rate equation
                if mcs > 100:
                    cell.targetVolume += 0.04 * concentration2 / (10 + concentration2)
                    cell.targetSurface += 0.12 * concentration2 / (10 + concentration2)

            # Hypoxic Cells
            if cell.type == 2:
                # print " #Hypoxic Volume: ", cell.volume
                pt = CompuCell.Point3D()
                pt.x = int(round(cell.xCM / max(float(cell.volume), 0.001)))
                pt.y = int(round(cell.yCM / max(float(cell.volume), 0.001)))
                pt.z = int(round(cell.zCM / max(float(cell.volume), 0.001)))
                concentration3 = fieldMalig.get(pt)
                # switch to Necrotic cell type
                if (concentration3 < self.necroticThresh and mcs > 100):
                    cell.type = 3
                # switch to Normal cell type
                if (concentration3 > self.nutrientThresh):
                    cell.type = 1

            # Necrotic Cells
            if cell.type == 3:
                # set growth rate equation
                cell.targetVolume -= 0.5
                cell.lambdaSurface = 0

    # self.output_file.write("\n")
    # #
    # #


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.doublingVolumeDict = None
        # # 0 - parent child position will be randomized between mitosis event
        # # negative integer - parent appears on the 'left' of the child
        # # positive integer - parent appears on the 'right' of the child
        # self.set_parent_child_position_flag(-1)

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
            # to change mitosis mode leave one of the below lines uncommented
            self.divide_cell_random_orientation(cell)
            # Other valid options
            # self.divide_cell_orientation_vector_based(cell,1,1,0)
            # self.divide_cell_along_major_axis(cell)
            # self.divide_cell_along_minor_axis(cell)

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
