import os
import random
from cc3d.core.PySteppables import *

random.seed()

# MaBoSS model details
bnd_file = os.path.join(os.path.dirname(__file__), "cellcycle.bnd")
cfg_file = os.path.join(os.path.dirname(__file__), "cellcycle_runcfg.cfg")
model_name = 'cell_cycle'
model_vars = ['Cdc20', 'cdh1', 'CycA', 'CycB', 'CycD', 'CycE', 'E2F', 'p27', 'Rb', 'UbcH10']
# Model parameters
target_volume = 36
lambda_volume = 2
time_tick = 0.2
time_step = time_tick / 10
discrete_time = False
# Data tracking
tracked_vars = model_vars.copy()  # Tracking cell states
track_pops = True  # Tracking fraction of states


class MaBoSSCellCycleSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.pop_win = None

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        # Put a MaBoSS cell cycle model into each cell
        cell_1 = None  # Temporary for sampling the MaBoSS model in a cell
        for cell in self.cell_list:
            self.add_maboss_to_cell(cell=cell,
                                    model_name=model_name,
                                    bnd_file=bnd_file,
                                    cfg_file=cfg_file,
                                    time_step=time_step,
                                    time_tick=time_tick,
                                    discrete_time=discrete_time,
                                    seed=random.randint(0, int(1E9)))
            # Set cc3d parameters
            cell.targetVolume = target_volume
            cell.lambdaVolume = lambda_volume
            if cell_1 is None:
                cell_1 = cell

        # Sample a cell and print its MaBoSS model to screen
        print(str(cell_1.maboss.cell_cycle.network))
        # Sample an initial node state
        print("Cell 1 beginning with CycD state=", str(cell_1.maboss.cell_cycle['CycD'].state))
        # Sample interactive MaBoSS run configuration
        print("Cell 1 has a MaBoSS engine      =", cell_1.maboss.cell_cycle)
        print("Cell 1 has a MaBoSS run config  =", cell_1.maboss.cell_cycle.run_config)

        # Track specified MaBoSS model variables
        for x in tracked_vars:
            self.track_cell_level_scalar_attribute(field_name=x, attribute_name=x)
        # Track population-level states
        if track_pops:
            self.pop_win = self.add_new_plot_window(title='States',
                                                    x_axis_title='Step',
                                                    y_axis_title='Fraction',
                                                    config_options={'legend': True})
            colors = ['blue', 'green', 'red', 'orange', 'yellow', 'white', 'pink', 'purple', 'cyan', 'brown']
            [self.pop_win.add_plot(model_vars[i], color=colors[i]) for i in range(len(model_vars))]

    def step(self, mcs):
        """
        Algorithm is as follows,

        - Step MaBoSS model in each cell
        - Update cell type and corresponding cell properties and selectively implement mitosis based on MaBoSS
          state vector in each cell
            - Cells in G1 are 50% bigger
            - Cells in G2 are 100% bigger
            - Cells in M divide

        :param mcs: current Monte Carlo step
        """
        # Step all MaBoSS models
        self.timestep_maboss()
        # Update cell types based on MaBoSS state vector
        pop_states = {k: 0 for k in model_vars}
        for cell in self.cell_list:
            cell.type = self.state_vector_to_cell_type(cell=cell)
            if cell.type == self.cell_type.CycleG1:
                cell.targetVolume = 1.5 * target_volume
            elif cell.type == self.cell_type.CycleS:
                pass
            elif cell.type == self.cell_type.CycleG2:
                cell.targetVolume = 2.0 * target_volume
            elif cell.type == self.cell_type.CycleM:
                self.divide_cell_random_orientation(cell=cell)
            else:
                cell.targetVolume = target_volume
            # Update cell state tracking data
            if tracked_vars or track_pops:
                for x in tracked_vars:
                    xstate = cell.maboss.cell_cycle[x].state
                    if tracked_vars:
                        cell.dict[x] = xstate
                    if track_pops:
                        pop_states[x] += int(xstate)
        # Update population state tracking data
        if track_pops:
            num_cells = len(self.cell_list)
            for x in model_vars:
                self.pop_win.add_data_point(x, mcs, float(pop_states[x] / num_cells))

    def update_attributes(self):
        # Reset parent cell cycle model to initial state and report rates during mitosis
        maboss_model = self.parent_cell.maboss.cell_cycle
        print(f"Parent cell {self.parent_cell.id} split with rates...")
        for x in model_vars:
            print(f"   {x} = ({maboss_model[x].rate_down}, {maboss_model[x].rate_up})")
            maboss_model[x].state = maboss_model[x].istate
        self.parent_cell.type = self.state_vector_to_cell_type(cell=self.parent_cell)
        # Clone attributes
        self.clone_parent_2_child()
        # Instantiate cell cycle model in child cell
        self.add_maboss_to_cell(cell=self.child_cell,
                                model_name=model_name,
                                bnd_file=bnd_file,
                                cfg_file=cfg_file,
                                time_step=time_step,
                                time_tick=time_tick,
                                discrete_time=discrete_time,
                                seed=random.randint(0, int(1E9)))
        # Ensure child cell cycle model is in the same state as that of the parent cell
        self.child_cell.maboss.cell_cycle.loadNetworkState(maboss_model.getNetworkState())

    def state_vector_to_cell_type(self, cell: CompuCell.CellG) -> int:
        """
        Maps the state vector of a cell according to MaBoSS to a cell type in CompuCell3D

        Type markers are as follows:

        - G1: CycD(+), Rb(-), E2F(+), CycA(-), p27(-), Cdc20(-), Cdh1(+), CycB(-)
        - S: CycD(+), Rb(-), CycE(+), CycA(+), p27(-), Cdc20(-), UbcH10(-), CycB(-)
        - G2: CycD(+), Rb(-), E2F(-), CycE(-), CycA(+), p27(-), Cdc20(-), Cdh1(-), UbcH10(+), CycB(+)
        - M: CycD(+), Rb(-), E2F(-), CycE(-), CycA(-), p27(-), Cdc20(+), Cdh1(+), UbcH10(+), CycB(-)

        :param cell: a cell
        :type cell: CompuCell.CellG
        :return: cell type label
        :rtype: int
        """
        state_vector = {x: cell.maboss.cell_cycle[x].state for x in model_vars}
        state_vector_g1 = {'CycD': True, 'Rb': False, 'E2F': True, 'CycA': False, 'p27': False,
                           'Cdc20': False, 'cdh1': True, 'CycB': False}
        state_vector_s = {'CycD': True, 'Rb': False, 'CycE': True, 'CycA': True, 'p27': False,
                          'Cdc20': False, 'UbcH10': False, 'CycB': False}
        state_vector_g2 = {'CycD': True, 'Rb': False, 'E2F': False, 'CycE': False, 'CycA': True, 'p27': False,
                           'Cdc20': False, 'cdh1': False, 'UbcH10': True, 'CycB': True}
        state_vector_m = {'CycD': True, 'Rb': False, 'E2F': False, 'CycE': False, 'CycA': False, 'p27': False,
                          'Cdc20': True, 'cdh1': True, 'UbcH10': True, 'CycB': False}
        if {k: state_vector[k] for k in state_vector_g1.keys()} == state_vector_g1:
            return self.cell_type.CycleG1
        elif {k: state_vector[k] for k in state_vector_s.keys()} == state_vector_s:
            return self.cell_type.CycleS
        elif {k: state_vector[k] for k in state_vector_g2.keys()} == state_vector_g2:
            return self.cell_type.CycleG2
        elif {k: state_vector[k] for k in state_vector_m.keys()} == state_vector_m:
            return self.cell_type.CycleM
        return self.cell_type.NonCycling

