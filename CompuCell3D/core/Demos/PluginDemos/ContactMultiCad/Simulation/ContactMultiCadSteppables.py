from cc3d.core.PySteppables import *
from random import random


class ContactMultiCadSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.table = None

    def set_type_contact_energy_table(self, _table):
        self.table = _table

    def start(self):

        c_multi_cad_data_accessor = self.contactMultiCadPlugin.getContactMultiCadDataAccessorPtr()

        for cell in self.cell_list:

            c_multi_cad_data = c_multi_cad_data_accessor.get(cell.extraAttribPtr)
            # jVec.set(0,self.table[cell.type])
            specificity_obj = self.table[cell.type]
            print("cell.type=", cell.type, " specificity_obj=", specificity_obj)
            if isinstance(specificity_obj, list):
                c_multi_cad_data.assignValue(0, (specificity_obj[1] - specificity_obj[0]) * random())
                c_multi_cad_data.assignValue(1, (specificity_obj[1] - specificity_obj[0]) * random())
            else:
                c_multi_cad_data.assignValue(0, specificity_obj)
                c_multi_cad_data.assignValue(1, specificity_obj / 2)
