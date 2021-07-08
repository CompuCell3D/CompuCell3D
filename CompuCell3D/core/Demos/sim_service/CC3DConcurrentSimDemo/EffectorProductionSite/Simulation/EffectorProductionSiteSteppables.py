import random

from cc3d.core.PySteppables import *
from cc3d.core.simservice import service_function

random.seed()

debug = False


class CellInitializerSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def start(self):

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting start!')

        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 5.0

            cell.dict['growth_delay'] = 0

        for y in range(0, self.dim.y):
            self.cell_field[0, y, 0].type = self.BASE


class GrowthSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):
        secretor = self.get_field_secretor("Signal")
        growth_threshold = 0.0025
        reduce_pr = 0.1

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        for cell in self.cell_list_by_type(self.PRODUCING):
            if mcs < cell.dict['growth_delay']:
                continue
            n_nbs = sum([1 for n, _ in self.get_cell_neighbor_data_list(cell) if n is not None and n.type == self.BASE])

            amount_seen = secretor.amountSeenByCell(cell)
            if amount_seen > growth_threshold and n_nbs > 0:
                cell.targetVolume += 1
            if cell.targetVolume > 25 and random.random() < reduce_pr:
                cell.targetVolume -= 1


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        split_volume = 50
        delay_time = 1000

        cells_to_divide = [cell for cell in self.cell_list_by_type(self.PRODUCING) if cell.volume > split_volume]
        [self.divide_cell_orientation_vector_based(cell, 1, 0, 0) for cell in cells_to_divide]
        for cell in cells_to_divide:
            cell.dict['growth_delay'] = mcs + delay_time

        if debug and len(cells_to_divide) > 0:
            print(f'EffectorProductionSite {self.__class__} growing {len(cells_to_divide)} cells.')

    def update_attributes(self):
        self.parent_cell.targetVolume /= 2.0
        self.clone_parent_2_child()
        self.child_cell.type = self.EFFECTOR


class ChemotaxisSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        chemotax_val = 5E4
        secretor = self.get_field_secretor("Signal")
        for cell in self.cell_list_by_type(self.EFFECTOR):
            cd = self.chemotaxisPlugin.getChemotaxisData(cell, "Signal")
            if cd is None:
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "Signal")
            concentration = secretor.amountSeenByCell(cell) / cell.volume
            cd.setLambda(chemotax_val / (1.0 + concentration))


class SignalHandlerSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.runBeforeMCS = 1

        self.total_signal = dict()

        # Signal values from the outside world can be deposited into this list via signal_receiver
        self._signal_buffer = list()
        service_function(self.signal_receiver)
        # For reporting total signal received
        self._total_received = 0
        service_function(self.total_signal_received)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):
        """
        Process received signals
        :param mcs:
        :return:
        """

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        # Process inputs from outside world
        self.total_signal[mcs] = 0
        for _val in self._signal_buffer:
            self.total_signal[mcs] += _val
        self._signal_buffer = list()
        self._total_received += self.total_signal[mcs]

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting {self.total_signal[mcs]} signal received.')

        delay_steps = 100
        if mcs - delay_steps in self.total_signal.keys():
            signal_val = self.total_signal.pop(mcs - delay_steps)
            # Test
            # signal_val = 1.0
            self.field.Signal[self.dim.x - 1, 0:self.dim.y, 0] = signal_val / self.dim.y

            if debug:
                print(f'EffectorProductionSite {self.__class__} reporting {signal_val} signal added.')

    def signal_receiver(self, val):
        self._signal_buffer.append(val)

    def total_signal_received(self):
        return self._total_received


class EffectorMigrationSignalSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        # Keeps a record of the cells that have left for a step
        # They can be withdrawn by the outside world via the transaction functions departed_cells and receive_cells
        self._cells_departed = 0
        service_function(self.departed_cells)
        service_function(self.receive_cells)
        # Keep a record of total cells sent
        self._total_cells_departed = 0
        service_function(self.total_cells_departed)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        exit_threshold = self.dim.x - 10

        self._cells_departed = 0
        for cell in self.cell_list_by_type(self.EFFECTOR):
            if cell.xCOM > exit_threshold:
                self._cells_departed += 1
                for ptd in self.get_cell_pixel_list(cell=cell):
                    self.cell_field[ptd.pixel.x, ptd.pixel.y, ptd.pixel.z] = None

        if self._cells_departed > 0:
            self._total_cells_departed += self._cells_departed
            if debug:
                print(f'EffectorProductionSite {self.__class__} has {self._cells_departed} cells available.')

    def departed_cells(self):
        return self._cells_departed

    def receive_cells(self, val):
        val = min(val, self._cells_departed)
        self._cells_departed -= val
        return val

    def total_cells_departed(self):
        return self._total_cells_departed
