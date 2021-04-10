from cc3d.core.PySteppables import *

import random

test_str = """
model MyModel()
-> A ; 0.01;
A = 0;
end
"""


class FocalPointPlasticityLinksSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

        # Let's see what links are up to concerning cell 1
        cell1 = self.fetch_cell_by_id(1)
        if cell1 is None:
            return

        # Let's see what cell 1 is linked to
        fpp_data_list = self.get_focal_point_plasticity_data_list(cell1)
        for link1j in self.get_fpp_links_by_cell(cell1):
            # Knowing the link and one linked cell, we can get the other linked cell
            cellj = link1j.getOtherCell(cell1)
            print(f'\nLinked cell id: {cellj.id}')

            # Or just knowing the link, we can get both linked cells
            initiator_cell, initiated_cell = link1j.cellPair
            print(f'Cell pair ids: {initiator_cell.id}, {initiated_cell.id}')

            # If data-handling is wrong, then the fetched data for this link might have multiple entries
            fppd = [fppd for fppd in fpp_data_list if fppd.neighborAddress.id == cellj.id]
            if len(fppd) != 1:
                print(fppd)
            assert len(fppd) == 1
            fppd = fppd[0]

            # Let's get link data via the old CC3D method and through new link-based methods
            print(f'Confirming linked cell by id: {cellj.id}, {fppd.neighborAddress.id}')
            print(f'lambda_distance: {link1j.getLambdaDistance()}, {fppd.lambdaDistance}')
            print(f'target_distance: {link1j.getTargetDistance()}, {fppd.targetDistance}')
            print(f'max_distance: {link1j.getMaxDistance()}, {fppd.maxDistance}')
            print(f'max_num_junctions: {link1j.getMaxNumberOfJunctions()}, {fppd.maxNumberOfJunctions}')
            print(f'activation_energy: {link1j.getActivationEnergy()}, {fppd.activationEnergy}')
            print(f'neighbor_order: {link1j.getNeighborOrder()}, {fppd.neighborOrder}')
            print(f'init_mcs: {link1j.getInitMCS()}, {fppd.initMCS}')

            # Let's check the link's derived properties
            print(f'length: {link1j.length}')
            print(f'tension: {link1j.tension}')

            # Links also have a general dictionary to store custom information!
            # Let's count how many times this link has been accessed
            my_var_name = 'times_counted'
            if my_var_name not in link1j.dict.keys():
                link1j.dict[my_var_name] = 0
            else:
                link1j.dict[my_var_name] += 1
            print(f'Number of times accessed: {link1j.dict[my_var_name]}')

            # Links can also have SBML, Antimony and CellML models
            if link1j.dict[my_var_name] == 0:
                self.add_antimony_to_link(link=link1j,
                                          model_string=test_str,
                                          model_name='myModel')
            A_val = link1j.sbml.myModel["A"]
            print(f'Current ODE variable value: {A_val}')

        self.timestep_sbml()

        # Now let's quickly get the list of all cells linked to cell 1
        linked_list = self.get_fpp_linked_cells(cell1)
        [print(f'Cell 1 has a link with cell {c.id}') for c in linked_list]

        # Let's calculate how many cells cell1 is linked to
        typej = {self.T1: self.T2, self.T2: self.T1}[cell1.type]
        n1j = self.get_number_of_fpp_junctions_by_type(cell1, typej)
        print(f'Cell {cell1.id} of type {cell1.type} has {n1j} junctions with type {typej}')

        # While we're at it, let's see how many links there are
        print(f'There are {self.get_number_of_fpp_links()} links')

        # Every 100 steps, let's vary the target distance of all links
        if mcs % 100 == 0:
            for link in self.get_focal_point_plasticity_link_list():
                max_distance = link.getMaxDistance()
                target_distance = max_distance * random.random()
                link.setMaxDistance(target_distance)

        # Every 1000 steps, let's try to manually form type 1 - type 1 and type 2 - type 2 links
        #   These links generally have a high activation energy, and so they won't form automatically
        #   However, we can make them form when we create them by setting the activation energy for a particular
        #   link to a permissible value
        #   Our rule will be that each cell can only have with link with a cell of the same type
        lambda_distance, target_distance, max_distance, activation_energy = 10, 5, 50, -50
        if mcs % 1000 == 0:
            for t in [self.T1, self.T2]:
                cell_list = [cell for cell in self.cell_list_by_type(t)]
                random.shuffle(cell_list)
                celli, cellj = cell_list[0:2]
                if self.get_number_of_fpp_junctions_by_type(celli, celli.type) == 0 and \
                        self.get_number_of_fpp_junctions_by_type(cellj, cellj.type) == 0:
                    new_link = self.new_fpp_link(celli, cellj, lambda_distance, target_distance, max_distance)
                    new_link.setActivationEnergy(activation_energy)

        # Every 1000 steps, let's randomly select a type 1 - type 2 link and break it
        if mcs % 1000 == 0:
            if self.get_number_of_fpp_links() > 0:
                link_list = [link for link in self.get_focal_point_plasticity_link_list()]
                random.shuffle(link_list)
                self.delete_fpp_link(link_list[0])

        # Every 10000 steps, let's break all links
        if mcs % 10000 == 0:
            [self.remove_all_cell_fpp_links(cell, links=True) for cell in self.cell_list]
