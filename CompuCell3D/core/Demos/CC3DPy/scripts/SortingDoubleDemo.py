"""
This example demonstrates how to specify, execute and visualize multiple
interactive CC3D simulations simultaneously in pure Python.
"""

from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
from cc3d.core.PyCoreSpecs import PottsCore, CellTypePlugin, VolumePlugin, BlobInitializer, ContactPlugin
from cc3d.core.simservice import service_cc3d

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"


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

    ####################################
    # Cell Distribution Initialization #
    ####################################
    # Initialize cells as a blob with a random distribution by type.
    blob_init_specs = BlobInitializer()
    blob_init_specs.region_new(width=5, radius=20, center=(50, 50, 0), cell_types=cell_types)
    specs.append(blob_init_specs)

    #########################
    # Differential Adhesion #
    #########################
    # Assign adhesion between cells by type such that cell sorting occurs,
    # but that the final arrangement of phenotypes is flipped between two specifications.
    contact_specs1 = ContactPlugin(neighbor_order=2)
    contact_specs1.param_new(type_1="Medium", type_2=cell_types[0], energy=20)
    contact_specs1.param_new(type_1="Medium", type_2=cell_types[1], energy=20)
    contact_specs1.param_new(type_1=cell_types[0], type_2=cell_types[0], energy=2)
    contact_specs1.param_new(type_1=cell_types[0], type_2=cell_types[1], energy=11)
    contact_specs1.param_new(type_1=cell_types[1], type_2=cell_types[1], energy=16)

    contact_specs2 = ContactPlugin(2)
    contact_specs2.param_new(type_1="Medium", type_2=cell_types[0], energy=20)
    contact_specs2.param_new(type_1="Medium", type_2=cell_types[1], energy=20)
    contact_specs2.param_new(type_1=cell_types[0], type_2=cell_types[0], energy=16)
    contact_specs2.param_new(type_1=cell_types[0], type_2=cell_types[1], energy=11)
    contact_specs2.param_new(type_1=cell_types[1], type_2=cell_types[1], energy=2)

    #####################
    # Simulation Launch #
    #####################
    # Initialize two CC3D simulation service instances and register all simulation specification.
    cc3d_sim1: CC3DSimService = service_cc3d()
    cc3d_sim1.register_specs(specs + [contact_specs1])
    cc3d_sim1.run()
    cc3d_sim1.init()
    cc3d_sim1.start()

    cc3d_sim2: CC3DSimService = service_cc3d()
    cc3d_sim2.register_specs(specs + [contact_specs2])
    cc3d_sim2.run()
    cc3d_sim2.init()
    cc3d_sim2.start()

    #################
    # Visualization #
    #################
    # Show a single frame to visualize simulation data as it is generated for each simulation.
    cc3d_sim1.visualize()
    cc3d_sim2.visualize()

    #############
    # Execution #
    #############

    # Wait for the user to trigger execution
    input('Press any key to continue...')

    # Execute 10k steps
    while cc3d_sim1.current_step < 10000:
        cc3d_sim1.step()
        cc3d_sim2.step()

    # Report performance
    print(cc3d_sim1.profiler_report)

    # Wait for the user to trigger termination
    input('Press any key to close...')


if __name__ == '__main__':
    main()
