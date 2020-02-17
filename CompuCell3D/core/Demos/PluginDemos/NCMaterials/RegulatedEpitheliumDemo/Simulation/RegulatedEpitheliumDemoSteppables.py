
from cc3d.core.PySteppables import *

import random

class DiffCell:
    def __init__(self, NCMaterialsPlugin, type_1, type_2):
        self.remod_type1 = {'AM': 0.0, 'BM': 0.0, 'CM': 1.0}
        self.remod_type2 = {'AM': 0.0, 'BM': 0.0, 'CM': 1.0}
        self.NCMaterialsPlugin = NCMaterialsPlugin
        
        self.remods = {}
        self.remods[type_1] = self.remod_type1
        self.remods[type_2] = self.remod_type2

    def diff_cell(self, cell, cell_type_int):
        cell.type = cell_type_int
        for comp, qty in self.remods[cell_type_int].items():
            self.NCMaterialsPlugin.setRemodelingQuantityByName(cell, comp, qty)


class RegulatedEpitheliumDemoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        self.diff_cell = DiffCell(self.NCMaterialsPlugin, self.TYPE1, self.TYPE2)
        
        self.map_str_int = {}
        self.map_str_int['Type1'] = self.TYPE1
        self.map_str_int['Type2'] = self.TYPE2

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        cell_response_list = self.get_ncmaterial_cell_response_list()
        if cell_response_list is None:
            return
        
        for this_cell, this_action, this_cell_type_diff in cell_response_list:
            if this_action == 'Proliferation':
                self.shared_steppable_vars['cells_to_divide'].append(this_cell)
            elif this_action == 'Differentiation':
                self.diff_cell.diff_cell(this_cell, self.map_str_int[this_cell_type_diff])

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
        

class DiffType2Steppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        self.diff_cell = DiffCell(self.NCMaterialsPlugin, self.TYPE1, self.TYPE2)

    def step(self, mcs):
        pr_t2 = 1e-04
        for cell in [cell for cell in self.cell_list if cell.type == self.TYPE2]:
            pr_diff = 0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if neighbor is not None and neighbor.type==self.TYPE2:
                    pr_diff += pr_t2*common_surface_area

            if pr_diff > random.random():
                self.diff_cell.diff_cell(cell, self.TYPE1)
        

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.set_parent_child_position_flag(0)

    def start(self):
        self.shared_steppable_vars['cells_to_divide'] = []

    def step(self, mcs):

        for cell in self.shared_steppable_vars['cells_to_divide']: # self.cells_to_divide:
            self.divide_cell_random_orientation(cell)
        
        self.shared_steppable_vars['cells_to_divide'] = []

    def update_attributes(self):

        self.clone_parent_2_child()


