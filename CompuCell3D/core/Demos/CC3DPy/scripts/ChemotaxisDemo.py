"""
This example demonstrates how to specify, execute and visualize an interactive
CC3D simulation of 2D chemotaxis in pure Python.
"""

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

from cc3d.core.PyCoreSpecs import (PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin, ChemotaxisPlugin,
                                   BlobInitializer)
from cc3d.core.PyCoreSpecs import PDEBOUNDARYFLUX, DiffusionSolverFE, ReactionDiffusionSolverFE
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService


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
    # Define four cell types called "T1" through "T4".
    cell_types = ["T1", "T2", "T3", "T4"]
    cell_type_specs = CellTypePlugin(*cell_types)
    specs.append(cell_type_specs)

    #####################
    # Volume Constraint #
    #####################
    # Assign a volume constraint to all cell types.
    target_volume, lambda_volume = 25, 2
    volume_specs = VolumePlugin()
    [volume_specs.param_new(ct, target_volume=target_volume, lambda_volume=lambda_volume) for ct in cell_types]
    specs.append(volume_specs)

    ############
    # Adhesion #
    ############
    # Assign uniform adhesion to all cells
    contact_specs = ContactPlugin(neighbor_order=2)
    for x1 in range(len(cell_types)):
        contact_specs.param_new(type_1="Medium", type_2=cell_types[x1], energy=16)
        for x2 in range(x1, len(cell_types)):
            contact_specs.param_new(type_1=cell_types[x1], type_2=cell_types[x2], energy=16)
    specs.append(contact_specs)

    ####################################
    # Cell Distribution Initialization #
    ####################################
    # Initialize cells as a blob with a random distribution by type.
    blob_init_specs = BlobInitializer()
    blob_init_specs.region_new(width=5, radius=20, center=(50, 50, 0), cell_types=cell_types)
    specs.append(blob_init_specs)

    #############
    # Diffusion #
    #############
    # Set up a diffusion field "F1" using DiffusionSolverFE.
    # Make the field have no flux conditions along the y-direction, value 0 along -x and value 1 along +x
    # Initialize the field with a steady state solution.
    # Use fluctuation compensator.
    f1_solver_specs = DiffusionSolverFE()  # Instantiate solver
    f1 = f1_solver_specs.field_new("F1")  # Declare a field for this solver
    f1.diff_data.decay_global = 1E-4  # Set global decay coefficient
    # Set type-specific diffusion and decay coefficients
    f1.diff_data.diff_types["Medium"] = 0.1
    for ct in cell_types:
        f1.diff_data.decay_types[ct] = 0.01
    # Set boundary conditions: Neumann on top and bottom, 0 on left, 1 on right
    f1.bcs.y_min_type = f1.bcs.y_max_type = PDEBOUNDARYFLUX
    f1.bcs.x_max_val = 1.0
    f1.diff_data.init_expression = "x / 100"  # Initialize with steady-state solution
    f1_solver_specs.fluc_comp = True  # Enable fluctuation compensator
    specs.append(f1_solver_specs)

    # Set up another diffusion field "F2" using ReactionDiffusionSolverFE.
    # Make the field have no flux conditions along the x-direction, value 0 along -y and value 1 along +y
    # Use fluctuation compensator.
    f2_solver_specs = ReactionDiffusionSolverFE()  # Instantiate solver
    f2 = f2_solver_specs.field_new("F2")  # Declare a field for this solver
    f2.diff_data.decay_global = 1E-4  # Set global decay coefficient
    # Set type-specific diffusion and decay coefficients
    f2.diff_data.diff_types["Medium"] = 0.2
    # Set boundary conditions: Neumann on left and right, 0 on bottom, 1 on top
    f2.bcs.x_min_type = f2.bcs.x_max_type = PDEBOUNDARYFLUX
    f2.bcs.y_max_val = 1.0
    f2_solver_specs.fluc_comp = True  # Enable fluctuation compensator
    specs.append(f2_solver_specs)

    ##############
    # Chemotaxis #
    ##############
    # Make two cell types chemotax along "F1", and the other two chemotax along "F2".
    # For both fields, make each cell type chemotax in the opposite direction.
    lambda_chemotaxis = 5E1
    chemotaxis_specs = ChemotaxisPlugin()
    cs = chemotaxis_specs.param_new("F1", "DiffusionSolverFE")
    cs.params_new(cell_types[0], lambda_chemotaxis)
    cs.params_new(cell_types[1], -lambda_chemotaxis)
    cs = chemotaxis_specs.param_new("F2", "ReactionDiffusionSolverFE")
    cs.params_new(cell_types[2], lambda_chemotaxis)
    cs.params_new(cell_types[3], -lambda_chemotaxis)
    specs.append(chemotaxis_specs)

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
    # Show a frame for the cell field and each diffusion field to visualize simulation data as it is generated.
    # Label the frame windows according to the displayed field.
    cc3d_sim.visualize()  # Default shows cells
    frame_f1 = cc3d_sim.visualize(name="F1")  # Include field name in window title
    frame_f2 = cc3d_sim.visualize(name="F2")  # Include field name in window title

    # Set the field to display
    frame_f1.field_name = "F1"
    frame_f2.field_name = "F2"

    # Set min and max values on fields
    frame_f1.min_range_fixed = frame_f1.max_range_fixed = True
    frame_f2.min_range_fixed = frame_f2.max_range_fixed = True
    frame_f1.min_range = frame_f2.min_range = 0.0
    frame_f1.max_range = frame_f2.max_range = 1.0

    # Issue draw to update visualization before execution
    frame_f1.draw()
    frame_f2.draw()

    # Wait for the user to trigger execution
    input('Press any key to continue...')

    # Execute 10k steps
    [cc3d_sim.step() for _ in range(10000)]

    # Wait for the user to trigger termination
    input('Press any key to close...')


if __name__ == '__main__':
    main()
