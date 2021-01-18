from cc3d.core.PySteppables import *
import random


class ContactLocalProductSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.table = None

    def set_type_contact_energy_table(self, table):
        self.table = table

    def start(self):
        random.seed(10)
        for cell in self.cell_list:
            specificity_obj = self.table[cell.type]
            if isinstance(specificity_obj, list):
                self.contactLocalProductPlugin.setJVecValue(cell, 0,
                                                            (specificity_obj[1] - specificity_obj[0]) * random.random())
            else:
                self.contactLocalProductPlugin.setJVecValue(cell, 0, specificity_obj)
