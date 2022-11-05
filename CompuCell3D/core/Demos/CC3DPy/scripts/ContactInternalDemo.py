"""
ContactInternal Plugin Demo

This example demonstrates how to specify cellular compartments, complex heterogeneous cells and adhesion on the
basis of cellular compartments.

Adjust the sliders to change the adhesion of cellular compartments between cells, as well as the adhesion of
cellular compartments within cells.
"""

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

from random import shuffle
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin, UniformInitializer, ContactLocalFlexPlugin,
                                   ContactInternalPlugin, PixelTrackerPlugin, FocalPointPlasticityPlugin)
from cc3d.core.PySteppables import SteppableBasePy


class ContactInternalDemoSteppable(SteppableBasePy):

    def start(self):
        """
        Initializes random intracellular configurations
        """

        specs_volume: VolumePlugin = self.specs.volume
        target_volume_A = specs_volume["A"].target_volume
        target_volume_B = specs_volume["B"].target_volume

        for cell in self.cell_list_by_type(self.cell_type.C):
            pixel_list = [px.pixel for px in self.get_cell_pixel_list(cell)]
            shuffle(pixel_list)
            pixel_list_A = [pixel_list.pop(x) for x in range(target_volume_A)]
            pixel_list_B = [pixel_list.pop(x) for x in range(target_volume_B)]
            cell_A = self.new_cell(self.cell_type.A)
            self.inventory.reassignClusterId(cell_A, cell.clusterId)
            cell_B = self.new_cell(self.cell_type.B)
            self.inventory.reassignClusterId(cell_B, cell.clusterId)
            for px in pixel_list_A:
                self.cell_field[px.x, px.y, px.z] = cell_A
            for px in pixel_list_B:
                self.cell_field[px.x, px.y, px.z] = cell_B


def main():
    ###############
    # Basic setup #
    ###############
    # An interactive CC3D simulation can be initialized from a list of core specs.
    # Start a list of core specs that define the simulation by specifying a two-dimensional simulation
    # with a 105x105 lattice and second-order Potts neighborhood, and three cells types "A", "B" and "C".
    # Also use PixelTrackerPlugin to quickly get cell locations during simulation.

    dim_x = dim_y = 105

    specs = [
        PottsCore(dim_x=dim_x,
                  dim_y=dim_y,
                  neighbor_order=2,
                  boundary_x="Periodic",
                  boundary_y="Periodic"),
        CellTypePlugin("A", "B", "C"),
        PixelTrackerPlugin()
    ]

    #####################
    # Volume Constraint #
    #####################
    # Assign a volume constraint to all cell types.
    volume_specs = VolumePlugin()
    volume_specs.param_new("A", lambda_volume=10, target_volume=10)
    volume_specs.param_new("B", lambda_volume=10, target_volume=10)
    volume_specs.param_new("C", lambda_volume=10, target_volume=30)
    specs.append(volume_specs)

    ############
    # Adhesion #
    ############
    # Assign uniform adhesion to all cells, and additional adhesion by cellular compartments
    contact_specs = ContactLocalFlexPlugin(neighbor_order=3)
    contact_specs.param_new(type_1="Medium", type_2="A", energy=30)
    contact_specs.param_new(type_1="Medium", type_2="B", energy=10)
    contact_specs.param_new(type_1="Medium", type_2="C", energy=20)
    contact_specs.param_new(type_1="A", type_2="A", energy=10)
    contact_specs.param_new(type_1="A", type_2="B", energy=20)
    contact_specs.param_new(type_1="A", type_2="C", energy=12)
    contact_specs.param_new(type_1="B", type_2="B", energy=15)
    contact_specs.param_new(type_1="B", type_2="C", energy=12)
    contact_specs.param_new(type_1="C", type_2="C", energy=20)
    specs.append(contact_specs)

    contact_intern_specs = ContactInternalPlugin(neighbor_order=3)
    contact_intern_specs.param_new(type_1="A", type_2="B", energy=30)
    contact_intern_specs.param_new(type_1="A", type_2="C", energy=5)
    contact_intern_specs.param_new(type_1="B", type_2="C", energy=5)
    specs.append(contact_intern_specs)

    ####################################
    # Cell Distribution Initialization #
    ####################################
    # Initialize cells over the entire domain.
    unif_init_specs = UniformInitializer()
    unif_init_specs.region_new(gap=0, width=7, pt_min=(0, 0, 0), pt_max=(dim_x, dim_y, 1), cell_types=["C"])
    specs.append(unif_init_specs)

    #########
    # Links #
    #########
    # Apply intercellular links between compartments of type "A",
    # and intracellular links between compartments "A" and "B"
    fpp_specs = FocalPointPlasticityPlugin()
    fpp_specs.params_new("A", "A", lambda_fpp=5, activation_energy=-50, target_distance=5, max_distance=20, max_junctions=2)
    fpp_specs.params_new("A", "B", lambda_fpp=5, activation_energy=-50, target_distance=10, max_distance=20, internal=True)
    specs.append(fpp_specs)

    #####################
    # Simulation Launch #
    #####################
    # Initialize a CC3D simulation service instance and register all simulation specification.
    cc3d_sim = CC3DSimService()
    cc3d_sim.register_specs(specs)
    cc3d_sim.register_steppable(steppable=ContactInternalDemoSteppable)
    cc3d_sim.run()
    cc3d_sim.init()
    cc3d_sim.start()

    #################
    # Visualization #
    #################
    # Show a single frame to visualize simulation data as it is generated and all steering widgets.
    # To better visualize cellular compartments, plot cluster boundaries instead of cell boundaries.
    frame = cc3d_sim.visualize()
    frame.cell_borders_on = False
    frame.cluster_borders_on = True
    frame.draw()

    #############
    # Execution #
    #############

    # Wait for the user to trigger execution
    input('Press any key to continue...')

    # Execute 10k steps
    while cc3d_sim.current_step < 10000:
        cc3d_sim.step()

    # Report performance
    print(cc3d_sim.profiler_report)

    # Wait for the user to trigger termination
    input('Press any key to close...')


if __name__ == '__main__':
    main()
