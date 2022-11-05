import os
from os.path import dirname, join
import random

from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell
from cc3d.core.simservice import service_function

random.seed()

debug = False


class InfectionSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'InfectionSite {self.__class__} reporting init!')

    def start(self):

        if debug:
            print(f'InfectionSite {self.__class__} reporting start!')

        frac_init_infected = 0.1
        cell_list = [c for c in self.cell_list]
        num_to_infect = int(len(cell_list) * frac_init_infected)
        random.shuffle(cell_list)
        for c in cell_list[0:num_to_infect]:
            c.type = self.INFECTING

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

        if debug:
            print(f'InfectionSite {self.__class__} reporting step!')

        pr_per_infecting = 1E-5
        for c in self.cell_list_by_type(self.SIGNALING):
            s = sum([1 for n, _ in self.get_cell_neighbor_data_list(c) if n is not None and n.type == self.INFECTING])
            if random.random() < s * pr_per_infecting:
                c.type = self.INFECTING

    def finish(self):
        return

    def on_stop(self):
        return


class SignalCountingSteppable(SteppableBasePy):
    """
    Keeps a record of the outbound signal in recent simulation history
    """
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        # For reporting total signal sent
        self._total_sent = 0
        service_function(self.total_signal_sent)

        if debug:
            print(f'InfectionSite {self.__class__} reporting init!')

    def step(self, mcs):
        """
        Report current signal value to the outside world
        :param mcs: current Monte Carlo step
        """

        if debug:
            print(f'InfectionSite {self.__class__} reporting step!')

        # Add signal to record
        secretor = self.get_field_secretor("SiteSignal")
        val = secretor.totalFieldIntegral()
        self.send_signal(val)
        self._total_sent += val

    def finish(self):
        return

    def on_stop(self):
        return

    def send_signal(self, val):
        recruitment_sim = self.shared_steppable_vars['recruitment_sim']
        recruitment_sim.signal_receiver(val)

    def total_signal_sent(self):
        return self._total_sent


class RecruitmentHandlerSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        self.cells_en_route = dict()
        # For reporting
        self._total_received = 0
        service_function(self.total_cells_received)

        if debug:
            print(f'InfectionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'InfectionSite {self.__class__} reporting step!')

        cells_migrating = self.get_all_migrating_cells()
        if cells_migrating > 0:
            if debug:
                print(f'InfectionSite {self.__class__} reporting {cells_migrating} migrating cells.')
            self.cells_en_route[mcs] = cells_migrating

        delay_steps = 200
        if mcs - delay_steps in self.cells_en_route.keys():
            cells_arriving = self.cells_en_route.pop(mcs - delay_steps)
            if debug and cells_arriving > 0:
                print(f'InfectionSite {self.__class__} reporting {cells_arriving} arriving cells.')
            for _ in range(cells_arriving):
                pt = CompuCell.Point3D()
                pt.x = 0
                pt.y = random.randint(0, self.dim.y - 8)
                pt.z = 0
                self.add_effector_cell(pt)

            self._total_received += cells_arriving

    def finish(self):
        return

    def on_stop(self):
        return

    def get_all_migrating_cells(self):
        recruitment_sim = self.shared_steppable_vars['recruitment_sim']
        departed_cells = recruitment_sim.departed_cells()
        received_cells = recruitment_sim.receive_cells(departed_cells)
        return received_cells

    def add_effector_cell(self, pt):
        """
        Add an effector cell that fights infection
        :param pt: {Point3D} bottom-left coordinate of seeding location
        :return:
        """
        cell = self.new_cell(self.EFFECTOR)
        cell_diameter = 7

        # Check inputs
        assert 0 <= pt.x < self.dim.x - cell_diameter and 0 <= pt.y < self.dim.y - cell_diameter

        # Build seed site
        self.cell_field[pt.x:pt.x + int(cell_diameter), pt.y:pt.y + int(cell_diameter), 0] = cell

    def total_cells_received(self):
        return self._total_received


class ChemotaxisSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'InfectionSite {self.__class__} reporting init!')

    def step(self, mcs):

        if debug:
            print(f'InfectionSite {self.__class__} reporting step!')

        chemotax_val = 5E4
        secretor = self.get_field_secretor("SiteSignal")
        for cell in self.cell_list_by_type(self.EFFECTOR):
            cd = self.chemotaxisPlugin.getChemotaxisData(cell, "SiteSignal")
            if cd is None:
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "SiteSignal")
            concentration = secretor.amountSeenByCell(cell) / cell.volume
            cd.setLambda(chemotax_val / (1.0 + concentration))


class KillingSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

        if debug:
            print(f'InfectionSite {self.__class__} reporting init!')

    def step(self, mcs):
        pr_death = 0.1

        if debug:
            print(f'InfectionSite {self.__class__} reporting step!')

        for cell in self.cell_list_by_type(self.INFECTING):
            n_nbs = sum([1 for n, _ in self.get_cell_neighbor_data_list(cell) if n is not None and n.type == self.EFFECTOR])
            if random.random() < pr_death * n_nbs:
                cell.targetVolume = 0.0


class EffectorProductionSiteSimHandler(CC3DServiceSteppableBasePy):
    def __init__(self, frequency=1):
        CC3DServiceSteppableBasePy.__init__(self, frequency)

        self.cc3d_sim_fname = join(dirname(dirname(dirname(__file__))),
                                   "EffectorProductionSite", "EffectorProductionSite.cc3d")
        self.cc3d_sim_output_dir = os.path.join(self.output_dir, "sim_output")

    def start(self):
        super().start()
        self.shared_steppable_vars['recruitment_sim'] = self.sim_service

        # Forward service functions from recruitment site sim
        service_function(self.sim_service.total_cells_departed)
        service_function(self.sim_service.total_signal_received)
