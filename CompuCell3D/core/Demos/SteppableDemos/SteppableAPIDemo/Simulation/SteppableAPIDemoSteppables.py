from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell


class SteppableAPIDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        self.set_max_mcs(20)

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        pt_1 = [10, 11, 4]
        pt_2 = [11, 15, 2]

        pt_hex_1 = self.cartesian_2_hex(pt_1)
        pt_1_cart = self.hex_2_cartesian(pt_hex_1)

        print('pt_1=', pt_1, ' pt_hex_1=', pt_hex_1, ' pt_1_cart=', pt_1_cart)

        pt_hex_2 = self.cartesian_2_hex(pt_2)
        pt_2_cart = self.hex_2_cartesian(pt_hex_2)
        print('pt_2=', pt_2, ' pt_hex_1=', pt_hex_2, ' pt_1_cart=', pt_2_cart)

        pt = CompuCell.Point3D(10, 12, 2)
        pt_np = self.point_3d_to_numpy(pt)

        print('pt=', pt, ' pt_np=', pt_np)

        pt_np = [12, 23, 32]
        pt = self.numpy_to_point_3d(pt_np)

        print('pt=', pt, ' pt_np=', pt_np)

        cell_1 = self.cell_field[30, 30, 0]
        cell_2 = self.cell_field[50, 50, 0]

        cells_different_flag = self.are_cells_different(cell1=cell_1, cell2=cell_2)

        print('cell_1=', cell_1, ' cell_2=', cell_2, ' cells_different_flag=', cells_different_flag)

        print('self.simulator.getNumSteps() = ', self.simulator.getNumSteps())

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
