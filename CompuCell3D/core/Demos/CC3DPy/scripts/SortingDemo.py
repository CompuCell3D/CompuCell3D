"""
This example demonstrates how to specify, execute and visualize an interactive
CC3D simulation of 2D cell sorting in pure Python.
"""

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
from cc3d.core.PyCoreSpecs import PottsCore, CellTypePlugin, VolumePlugin, BlobInitializer, ContactPlugin


def main():
    ###############
    # Basic setup #
    ###############
    # An interactive CC3D simulation can be initialized from a list of core specs.
    # Start a list of core specs that define the simulation by specifying a two-dimensional simulation
    # with a 100x100 lattice and second-order Potts neighborhood.
    specs = [PottsCore(dim_x=100, dim_y=100, neighbor_order=2)]

    ##############
    # Cell Types #
    ##############
    # Define two cell types called "Condensing" and "NonCondensing".
    cell_types = ["Condensing", "NonCondensing"]
    specs.append(CellTypePlugin(*cell_types))

    #####################
    # Volume Constraint #
    #####################
    # Assign a volume constraint to both cell types.
    target_volume, lambda_volume = 25, 2
    volume_specs = VolumePlugin()
    volume_specs.param_new(cell_types[0], target_volume=target_volume, lambda_volume=lambda_volume)
    volume_specs.param_new(cell_types[1], target_volume=target_volume, lambda_volume=lambda_volume)
    specs.append(volume_specs)

    #########################
    # Differential Adhesion #
    #########################
    # Assign adhesion between cells by type such that cell sorting occurs.
    contact_specs = ContactPlugin(neighbor_order=2)
    contact_specs.param_new(type_1="Medium", type_2=cell_types[0], energy=20)
    contact_specs.param_new(type_1="Medium", type_2=cell_types[1], energy=20)
    contact_specs.param_new(type_1=cell_types[0], type_2=cell_types[0], energy=2)
    contact_specs.param_new(type_1=cell_types[0], type_2=cell_types[1], energy=11)
    contact_specs.param_new(type_1=cell_types[1], type_2=cell_types[1], energy=16)
    specs.append(contact_specs)

    ####################################
    # Cell Distribution Initialization #
    ####################################
    # Initialize cells as a blob with a random distribution by type.
    blob_init_specs = BlobInitializer()
    blob_init_specs.region_new(width=5, radius=20, center=(50, 50, 0), cell_types=cell_types)
    specs.append(blob_init_specs)

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
