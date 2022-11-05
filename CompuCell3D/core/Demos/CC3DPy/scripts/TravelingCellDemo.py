"""
This example demonstrates how to build and execute multiple CC3D simulations to simulate
multiple interacting sites in a biological system.

In this simple example, cells chemotax through, and to, multiple locations.
Each location
    - is simulated using a separate CC3D simulation,
    - has an inlet boundary, where cells enters,
    - has an outlet boundary, where cells leave,
    - is separated from every other location by a distance that takes time to traverse by cells
"""

import itertools
import random
from typing import Dict, List
from cc3d.core.simservice import service_cc3d, service_function
from cc3d.core.PySteppables import SteppableBasePy
from cc3d.core import PyCoreSpecs
from cc3d.cpp import CompuCell

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

num_sites = 4
"""Number of locations to simulate"""

travel_steps = 100
"""Number of simulation steps required to get from one location to another"""

num_cells_per_site = 1
"""Number of initial cells in each location"""

num_steps = 10000
"""Number of time steps to simulate"""

cell_type_name = 'Traveler'
"""Name of cell type"""

cell_length_target = 6
"""Target length of cell type"""

cell_chemo_lm = 500
"""Chemotaxis lambda parameter"""

field_name = 'Chemoattractant'
"""Name of field along which cells chemotax"""

border_len = 6
"""Boundary of locations demarcating inlet and outlet zones"""


class LocationInfo:
    """
    Simple data container for tracking where all a traveling cell travels.

    For each location where a cell travels, an instance of this class records the location, time spent and local cell id
    """

    cell_dict_key = 'location_info'
    """Key for uniform storage in cell dictionary"""

    def __init__(self, local_id: int, location_id: int, entry_time: int, exit_time: int = None):

        self.local_id = local_id
        self.location_id = location_id
        self.entry_time = entry_time
        self.exit_time = exit_time


class LocationSteppable(SteppableBasePy):
    """
    A steppable that implements the I/O of a location.

    Each location can have cells, and those cells can leave the location by reaching the +X boundary of the domain.
    When a cell "leaves", the steppable places the cell in a queue from which a controlling process can pull
    and place departed cells in other locations. To this end, the steppable also handles placement of cells into
    the location as specified by a controlling process, and places the cells along the -X boundary.
    All methods are made available on a CC3D simulation instance using simservice service functions.
    """

    def __init__(self, frequency: int = 1):

        super().__init__(frequency=frequency)

        self._location_id = -1
        """Unique id of location"""

        self._outlet_queue: List[LocationInfo] = []
        """List of all cell location info that have reached the outlet and are ready to depart"""

    def start(self):

        # Add methods to simulation instance interface
        service_function(self.cells_incoming)
        service_function(self.take_outgoing)
        service_function(self.set_location_id)
        service_function(self.do_cell_init)

    def step(self, mcs):

        # Check for cells at the outlet
        for cell in self.cell_list:
            cell: CompuCell.CellG
            if cell.xCOM > self.dim.x - border_len:
                # Add cell to the output queue, so that other locations can grab it
                cell.dict[LocationInfo.cell_dict_key][-1].exit_time = mcs
                self._outlet_queue.append(cell.dict[LocationInfo.cell_dict_key])

                # Remove cell from this location
                px_list = self.get_cell_pixel_list(cell)
                for px in px_list:
                    self.cell_field[px.pixel.x, px.pixel.y, px.pixel.z] = None

    @property
    def traveler_type(self):
        """Convenience property to identify the Traveler cell type"""
        return getattr(self.cell_type, cell_type_name)

    def set_location_id(self, _id: int):
        """Sets the unique id of the location"""
        self._location_id = _id

    def do_cell_init(self, num_init: int):
        """
        Initializes cells in the location

        :param num_init: number of cells to initialize
        """

        for _ in range(num_init):
            new_cell = self.new_cell(self.traveler_type)
            new_cell.dict[LocationInfo.cell_dict_key] = [LocationInfo(new_cell.id, self._location_id, 0)]

            pos_x = random.randint(border_len, self.dim.x - cell_length_target - border_len)
            pos_y = random.randint(0, self.dim.y - cell_length_target)

            for x, y in itertools.product(range(cell_length_target), range(cell_length_target)):
                self.cell_field[pos_x + x, pos_y + y, 0] = new_cell

    def cells_incoming(self, location_info: List[LocationInfo]):
        """
        Add cells at the inlet

        :param location_info: location info of each incoming cell
        """

        for info in location_info:
            new_cell = self.new_cell(self.traveler_type)
            new_cell.dict['location_info'] = info + [LocationInfo(new_cell.id, self._location_id, self.mcs)]

            pos_y = random.randint(0, self.dim.y - cell_length_target)
            for x, y in itertools.product(range(cell_length_target), range(cell_length_target)):
                self.cell_field[cell_length_target + x, pos_y + y, 0] = new_cell

    def num_outgoing(self):
        """Number of outgoing cells"""
        return len(self._outlet_queue)

    def take_outgoing(self) -> List[LocationInfo]:
        """Take all outgoing cells"""
        num_taken = len(self._outlet_queue)
        return [self._outlet_queue.pop(0) for _ in range(num_taken)]


def main():

    specs = [
        PyCoreSpecs.PottsCore(dim_x=100, dim_y=100),
        PyCoreSpecs.CellTypePlugin(cell_type_name),
        PyCoreSpecs.PixelTrackerPlugin(),
        PyCoreSpecs.CenterOfMassPlugin()
    ]

    volume_specs = PyCoreSpecs.VolumePlugin()
    volume_specs.param_new(cell_type_name, target_volume=cell_length_target * cell_length_target, lambda_volume=2.0)
    specs.append(volume_specs)

    contact_specs = PyCoreSpecs.ContactPlugin(neighbor_order=2)
    contact_specs.param_new("Medium", cell_type_name, 10)
    contact_specs.param_new(cell_type_name, cell_type_name, 20)
    specs.append(contact_specs)

    diff_solver_specs = PyCoreSpecs.DiffusionSolverFE()
    field_specs = diff_solver_specs.field_new(field_name)
    field_specs.bcs.x_min_type = field_specs.bcs.x_max_type = "Value"
    field_specs.bcs.x_min_val = 0.0
    field_specs.bcs.x_max_val = 1.0
    field_specs.diff_data.diff_global = 0.1
    field_specs.diff_data.init_expression = 'x / 100'
    specs.append(diff_solver_specs)

    chemotaxis_specs = PyCoreSpecs.ChemotaxisPlugin()
    chemotaxis_type_specs = chemotaxis_specs.param_new(field_name=field_name, solver_name=diff_solver_specs.name)
    chemotaxis_type_specs.params_new(cell_type_name, cell_chemo_lm)
    specs.append(chemotaxis_specs)

    sims = []
    next_site_ids = []
    for i in range(num_sites):
        print(f'Initializing site {i}')

        next_site_ids.append(i + 1)
        cc3d_sim = service_cc3d()
        cc3d_sim.register_steppable(steppable=LocationSteppable)
        cc3d_sim.register_specs(specs)

        cc3d_sim.run()
        cc3d_sim.init()
        cc3d_sim.start()

        cc3d_sim.visualize(name=f'Location {i}')

        cc3d_sim.set_location_id(i)
        cc3d_sim.do_cell_init(num_cells_per_site)

        sims.append(cc3d_sim)
    next_site_ids[-1] = 0

    input('Press any key to continue...')

    queued_info: Dict[int, List[List[LocationInfo]]] = {}

    for current_step in range(num_steps):
        departed_cells = []
        for cc3d_sim in sims:
            cc3d_sim.step()
            departed_cells.append(cc3d_sim.take_outgoing())

        queued_info[current_step + travel_steps] = departed_cells
        try:
            incoming_cells = queued_info.pop(current_step)
        except KeyError:
            incoming_cells = []

        for i, dc in enumerate(incoming_cells):
            if dc:
                sims[next_site_ids[i]].cells_incoming(dc)

    input('Press any key to continue...')


if __name__ == '__main__':
    main()
