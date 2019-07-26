from random import random
from cc3d.core.PySteppables import *


class VolumeConstraintSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0
            cell.dict['lineage_list'] = []

    def step(self, mcs):
        fgf = self.field.FGF

        for cell in self.cell_list:
            if cell.type == self.CONDENSING and mcs < 1500:
                # Condensing cell
                concentration = fgf[int(round(cell.xCOM)), int(round(cell.yCOM)), int(round(cell.zCOM))]
                # increase cell's target volume
                cell.targetVolume += 0.1 * concentration

            if mcs > 1500:
                # removing all cells
                cell.targetVolume -= 1
                # increase cell's target volume
                if cell.targetVolume < 0.0:
                    cell.targetVolume = 0.0


# MItosis data has to have base
# class "object" otherwise if cell will be deleted CC3D may crash due to improper garbage collection
class MitosisData(object):
    def __init__(self, mcs=-1, parent_id=-1, parent_type=-1, offspring_id=-1, offspring_type=-1):
        self.MCS = mcs
        self.parentId = parent_id
        self.parentType = parent_type
        self.offspringId = offspring_id
        self.offspringType = offspring_type

    def __str__(self):
        return "Mitosis time=" + str(self.MCS) + " parentId=" + str(self.parentId) + " offspringId=" + str(
            self.offspringId)


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def step(self, mcs):
        cells_to_divide = []
        for cell in self.cell_list:
            if cell.volume > 50:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0
        self.clone_parent_2_child()

        if random() < 0.5:
            self.child_cell.type = self.CONDENSINGDIFFERENTIATED

        # will record mitosis data in parent and offspring cells
        mcs = self.simulator.getStep()
        mit_data = MitosisData(mcs, self.parent_cell.id, self.parent_cell.type, self.child_cell.id,
                               self.child_cell.type)
        self.parent_cell.dict['lineage_list'].append(mit_data)
        self.child_cell.dict['lineage_list'].append(mit_data)


class MitosisDataPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cellList:
            mit_data_list = cell.dict
            if len(mit_data_list) > 0:
                print("MITOSIS DATA FOR CELL ID", cell.id)
                for mit_data in mit_data_list:
                    print(mit_data)
