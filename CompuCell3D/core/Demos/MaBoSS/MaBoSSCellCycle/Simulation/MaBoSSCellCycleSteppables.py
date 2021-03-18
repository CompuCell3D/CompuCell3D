import os
import random

from cc3d.core.PySteppables import *

random.seed()

bnd_file = os.path.join(os.path.dirname(__file__), "cellcycle.bnd")
cfg_file = os.path.join(os.path.dirname(__file__), "cellcycle_runcfg.cfg")
model_name = 'cell_cycle'
model_vars = ['Cdc20', 'cdh1', 'CycA', 'CycB', 'CycD', 'CycE', 'E2F', 'p27', 'Rb', 'UbcH10']

# Model parameters
target_volume = 36
lambda_volume = 2
time_tick = 0.2
time_step = time_tick / 100
discrete_time = False


class MaBoSSCellCycleSteppable(MitosisSteppableBase):

    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        # Put a MaBoSS cell cycle model into each cell
        cell_1 = None
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

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        # Step all MaBoSS models
        self.timestep_maboss()
        # Update cell types based on MaBoSS state vector
        for cell in self.cell_list:
            if cell.maboss.cell_cycle['CycB'].state:
                # If CycB is high, then the cell is in G2
                cell.type = self.CYCLEG2
                cell.targetVolume = 2.0 * target_volume
            elif cell.maboss.cell_cycle['E2F'].state and not cell.maboss.cell_cycle['CycA'].state:
                # If E2F is high and CycA is low, then the cell is in G1
                cell.type = self.CYCLEG1
                cell.targetVolume = 1.5 * target_volume
            elif cell.maboss.cell_cycle['CycE'].state and cell.maboss.cell_cycle['CycA'].state:
                # If CycE is high and CycA is high, then the cell is in S
                cell.type = self.CYCLES
            elif cell.maboss.cell_cycle['Cdc20'].state:
                # If Cdc20 is high and CycB is low, then the cell is in M
                cell.type = self.CYCLEM
                self.divide_cell_random_orientation(cell=cell)
            else:
                # Otherwise, the cell is not cycling
                cell.type = self.NONCYCLING
                cell.targetVolume = target_volume

    def update_attributes(self):
        self.clone_parent_2_child()
        # Reset parent cell cycle model to initial state and report rates during mitosis
        maboss_model = self.parent_cell.maboss.cell_cycle
        print(f"Parent cell {self.parent_cell.id} split with rates...")
        for x in model_vars:
            print(f"   {x} = ({maboss_model[x].rate_down}, {maboss_model[x].rate_up})")
            maboss_model[x].state = maboss_model[x].istate
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
        maboss_model_child = self.child_cell.maboss.cell_cycle
        maboss_model_child.loadNetworkState(maboss_model.getNetworkState())
