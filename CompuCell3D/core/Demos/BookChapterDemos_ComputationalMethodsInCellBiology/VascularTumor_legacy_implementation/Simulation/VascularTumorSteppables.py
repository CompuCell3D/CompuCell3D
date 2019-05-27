from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1, ):
        SteppableBasePy.__init__(self, frequency)
        self.fieldNameVEGF2 = 'VEGF2'
        self.fieldNameGlucose = 'Glucose'

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
        fieldVEGF2 = CompuCell.getConcentrationField(self.simulator, self.fieldNameVEGF2)
        fieldGlucose = CompuCell.getConcentrationField(self.simulator, self.fieldNameGlucose)

        for cell in self.cellList:
            # print cell.volume
            # NeoVascular
            if cell.type == self.NEOVASCULAR:
                totalArea = 0
                # pt=CompuCell.Point3D()
                # pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                # pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                # pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))

                # VEGFConcentration=fieldVEGF2.get(pt)

                VEGFConcentration = fieldVEGF2[int(round(cell.xCOM)), int(round(cell.yCOM)), int(round(cell.zCOM))]

                # cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
                neighbor_list = self.get_cell_neighbor_data_list(cell)
                for neighbor, common_surface_area in neighbor_list:
                    # Check to ensure cell neighbor is not medium
                    if neighbor:
                        if neighbor.type == self.VASCULAR or neighbor.type == self.NEOVASCULAR:
                            # sum up common surface area of cell with its neighbors
                            totalArea += common_surface_area
                            # print "  commonSurfaceArea:",neighborSurfaceData.commonSurfaceArea
                # print totalArea
                if totalArea < 45:
                    # Growth rate equation

                    cell.targetVolume += 2.0 * VEGFConcentration / (0.01 + VEGFConcentration)
                    print("totalArea", totalArea, "cell growth rate: ",
                          2.0 * VEGFConcentration / (0.01 + VEGFConcentration), "cell Volume: ", cell.volume)

            # Proliferating Cells
            if cell.type == self.PROLIFERATING:

                # pt=CompuCell.Point3D()
                # pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
                # pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
                # pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
                # GlucoseConcentration=fieldGlucose.get(pt)

                GlucoseConcentration = fieldGlucose[int(round(cell.xCOM)), int(round(cell.yCOM)), int(round(cell.zCOM))]

                # Proliferating Cells become Necrotic when GlucoseConcentration is low
                if GlucoseConcentration < 0.001 and mcs > 1000:
                    cell.type = self.NECROTIC
                    # set growth rate equation -- fastest cell cycle is 24hours or 1440 mcs
                    # --- 32voxels/1440mcs= 0.022 voxel/mcs
                cell.targetVolume += 0.022 * GlucoseConcentration / (0.05 + GlucoseConcentration)
                # print( "growth rate: ", 0.044*GlucoseConcentration/(0.05 + GlucoseConcentration),
                # "GlucoseConcentration", GlucoseConcentration)

            # Necrotic Cells
            if cell.type == self.NECROTIC:
                # sNecrotic Cells shrink at a constant rate
                cell.targetVolume -= 0.1


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def step(self, mcs):

        cells_to_divide = []

        for cell in self.cellList:
            if cell.type == self.PROLIFERATING and cell.volume > 64:
                cells_to_divide.append(cell)
            if cell.type == self.NEOVASCULAR and cell.volume > 128:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divideCellRandomOrientation(cell)

    def updateAttributes(self):
        self.parentCell.targetVolume /= 2.0  # reducing parent target volume
        self.cloneParent2Child()
