
from cc3d.core.PySteppables import *

class GauntletGameDemoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

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
                if len(this_cell_type_diff) == 0:
                    self.shared_steppable_vars['cells_to_divide'].append(this_cell)
                else: 
                    self.shared_steppable_vars['cells_to_divide_asym'].append(this_cell)
                        
            elif this_action == 'Death':
                this_cell.type = self.TYPE3
            elif this_action == 'Differentiation':
                if this_cell_type_diff == 'Type1':
                    this_cell.type = self.TYPE1
                elif this_cell_type_diff == 'Type2':
                    this_cell.type = self.TYPE2
        

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.set_parent_child_position_flag(0)

    def start(self):
        self.shared_steppable_vars['cells_to_divide'] = []

    def step(self, mcs):

        for cell in self.shared_steppable_vars['cells_to_divide']:
            self.divide_cell_random_orientation(cell)
        
        self.shared_steppable_vars['cells_to_divide'] = []

    def update_attributes(self):

        self.clone_parent_2_child()

class MitosisSteppableAsym(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.set_parent_child_position_flag(0)

    def start(self):
        self.shared_steppable_vars['cells_to_divide_asym'] = []

    def step(self, mcs):
        
        print(self.shared_steppable_vars)
        
        for cell in self.shared_steppable_vars['cells_to_divide_asym']:
            self.divide_cell_random_orientation(cell)
        
        self.shared_steppable_vars['cells_to_divide_asym'] = []

    def update_attributes(self):

        self.clone_parent_2_child()
        self.child_cell.type = self.TYPE2
