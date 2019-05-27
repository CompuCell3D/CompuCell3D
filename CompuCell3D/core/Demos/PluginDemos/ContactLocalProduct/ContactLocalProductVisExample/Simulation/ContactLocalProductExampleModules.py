from cc3d.core.PySteppables import *
from random import random


class ContactLocalProductSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.table = None

    def set_type_contact_energy_table(self, table):
        self.table = table

    def start(self):
        for cell in self.cell_list:
            specificity_obj = self.table[cell.type]
            if isinstance(specificity_obj, list):
                self.contactLocalProductPlugin.setJVecValue(cell, 0, (specificity_obj[1] - specificity_obj[0]) * random())
            else:
                self.contactLocalProductPlugin.setJVecValue(cell, 0, specificity_obj)


class ContactSpecVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_py('ContSpec')

    def step(self, mcs):

        conc_spec_field = self.field.ContSpec
        for x, y, z in self.every_pixel():

            cell = self.cell_field[x, y, z]
            if cell:
                conc_spec_field[x, y, z] = self.contactLocalProductPlugin.getJVecValue(cell, 0)
            else:
                conc_spec_field[x, y, z] = 0.0
