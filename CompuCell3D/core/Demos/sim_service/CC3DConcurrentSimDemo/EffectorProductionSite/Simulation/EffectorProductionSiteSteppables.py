import random

from cc3d.core.PySteppables import *

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

        if len(cells_to_divide) > 0:
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

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def start(self):
        """
        Expose signal receiver to the outside world
        :return:
        """

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting start!')

        pg = CompuCellSetup.persistent_globals
        if not isinstance(pg.input_object, dict):
            pg.input_object = dict()
        if 'SignalHandlerSteppable' not in pg.input_object.keys():
            pg.input_object['SignalHandlerSteppable'] = dict()
        if 'receive_signal' not in pg.input_object['SignalHandlerSteppable']:
            pg.input_object['SignalHandlerSteppable']['receive_signal'] = list()

    def step(self, mcs):
        """
        Process received signals
        :param mcs:
        :return:
        """

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        # Process inputs from outside world
        pg = CompuCellSetup.persistent_globals
        self.total_signal[mcs] = 0
        for _val in pg.input_object['SignalHandlerSteppable']['receive_signal']:
            self.total_signal[mcs] += _val
        pg.input_object['SignalHandlerSteppable']['receive_signal'] = list()

        print(f'EffectorProductionSite {self.__class__} reporting {self.total_signal[mcs]} signal received.')

        delay_steps = 100
        if mcs - delay_steps in self.total_signal.keys():
            signal_val = self.total_signal.pop(mcs - delay_steps)
            # Test
            # signal_val = 1.0
            self.field.Signal[self.dim.x - 1, 0:self.dim.y, 0] = signal_val / self.dim.y

            print(f'EffectorProductionSite {self.__class__} reporting {signal_val} signal added.')


class EffectorMigrationSignalSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'EffectorProductionSite {self.__class__} reporting step!')

        exit_threshold = self.dim.x - 10
        
        cells_leaving = 0
        for cell in self.cell_list_by_type(self.EFFECTOR):
            if cell.xCOM > exit_threshold:
                cells_leaving += 1
                for ptd in self.get_cell_pixel_list(cell=cell):
                    self.cell_field[ptd.pixel.x, ptd.pixel.y, ptd.pixel.z] = None

        pg = CompuCellSetup.persistent_globals
        if not isinstance(pg.return_object, dict):
            pg.return_object = dict()
        if 'EffectorMigrationSignalSteppable' not in pg.return_object.keys():
            pg.return_object['EffectorMigrationSignalSteppable'] = dict()
        pg.return_object['EffectorMigrationSignalSteppable']['cells_leaving'] = cells_leaving

        if cells_leaving > 0:
            print(f'EffectorProductionSite {self.__class__} sending {cells_leaving} cells.')
