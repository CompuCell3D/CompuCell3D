"""
This example demonstrates how to specify cell adhesion on the basis of molecular species.
"""

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin
from cc3d.core.PyCoreSpecs import UniformInitializer
from cc3d.core.PyCoreSpecs import AdhesionFlexPlugin
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService


def main():
    ###############
    # Basic setup #
    ###############
    # An interactive CC3D simulation can be initialized from a list of core specs.
    # Start a list of core specs that define the simulation by specifying a two-dimensional simulation
    # with a 100x100 lattice and second-order Potts neighborhood, and metadata to use multithreading
    dim_x = dim_y = 100
    specs = [
        Metadata(num_processors=4),
        PottsCore(dim_x=dim_x,
                  dim_y=dim_y,
                  neighbor_order=2,
                  boundary_x="Periodic",
                  boundary_y="Periodic")
    ]

    ##############
    # Cell Types #
    ##############
    # Define three cell types called "T1" through "T3".
    cell_types = ["T1", "T2", "T3"]
    specs.append(CellTypePlugin(*cell_types))

    #####################
    # Volume Constraint #
    #####################
    # Assign a volume constraint to all cell types.
    volume_specs = VolumePlugin()
    for ct in cell_types:
        volume_specs.param_new(ct, target_volume=25, lambda_volume=2)
    specs.append(volume_specs)

    ############
    # Adhesion #
    ############
    # Assign uniform adhesion to all cells, and additional adhesion by molecular species
    contact_specs = ContactPlugin(neighbor_order=2)
    for idx1 in range(len(cell_types)):
        contact_specs.param_new(type_1="Medium", type_2=cell_types[idx1], energy=16)
        for idx2 in range(idx1, len(cell_types)):
            contact_specs.param_new(type_1=cell_types[idx1], type_2=cell_types[idx2], energy=16)
    specs.append(contact_specs)

    adhesion_specs = AdhesionFlexPlugin(neighbor_order=2)
    adhesion_specs.density_new(molecule="M1", cell_type="T1", density=1.0)
    adhesion_specs.density_new(molecule="M2", cell_type="T2", density=1.0)
    formula = adhesion_specs.formula_new()
    formula.param_set("M1", "M1", -10.0)
    formula.param_set("M1", "M2", 0.0)
    formula.param_set("M2", "M2", 10.0)
    specs.append(adhesion_specs)

    ####################################
    # Cell Distribution Initialization #
    ####################################
    # Initialize cells over the entire domain.
    unif_init_specs = UniformInitializer()
    unif_init_specs.region_new(width=5, pt_min=(0, 0, 0), pt_max=(dim_x, dim_y, 1),
                               cell_types=["T1", "T1", "T2", "T2", "T3"])
    specs.append(unif_init_specs)

    #####################
    # Simulation Launch #
    #####################
    # Initialize a CC3D simulation service instance and register all simulation specification.
    cc3d_sim = CC3DSimService()
    cc3d_sim.register_specs(specs)
    cc3d_sim.run()
    cc3d_sim.init()
    cc3d_sim.start()

    #################
    # Visualization #
    #################
    # Show a single frame to visualize simulation data as it is generated.
    cc3d_sim.visualize()

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
